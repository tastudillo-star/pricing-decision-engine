import mysql.connector
from mysql.connector import Error
from typing import Iterable, List, Tuple, Dict, Optional, Callable, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import streamlit as st
from utils.auth import Auth


HOST = st.secrets["HOST"]
USER = st.secrets["USER"]
PASSWORD = st.secrets["PASSWORD"]
DATABASE = st.secrets["DATABASE"]

class MySQLBulkLoader:
    """
    Cargador masivo tolerante para MySQL:
    - Inserta DataFrame por lotes con ON DUPLICATE KEY UPDATE.
    - Salta filas malas sin detener la carga.
    - Reporta estadísticas y ejemplos de filas malas al final.
    """

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 3306,
        connect_timeout: int = 10,
    ):
        self._conn_cfg = {
            "host": host,
            "user": user,
            "password": password,
            "database": database,
            "port": port,
            "connection_timeout": connect_timeout,
            "autocommit": False,
        }

    def _get_connection(self):
        return mysql.connector.connect(**self._conn_cfg)

    def _set_optimizations(self, cursor, enable: bool):
        if enable:
            cursor.execute("SET unique_checks=0;")
            cursor.execute("SET foreign_key_checks=0;")
        else:
            cursor.execute("SET unique_checks=1;")
            cursor.execute("SET foreign_key_checks=1;")

    def _build_insert_sql(self, table_name: str, columns: List[str]) -> str:
        cols_csv = ",".join(columns)
        placeholders = ",".join(["%s"] * len(columns))
        assignments = ", ".join([f"{c}=VALUES({c})" for c in columns])
        return (
            f"INSERT INTO {table_name} ({cols_csv}) "
            f"VALUES ({placeholders}) "
            f"ON DUPLICATE KEY UPDATE {assignments}"
        )

    def _pythonize_value(self, v, coerce_na_to_none: bool):
        if coerce_na_to_none and pd.isna(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, pd.Timestamp):
            return v.to_pydatetime()
        return v

    def _prepare_batch_rows(self, raw_rows: List[Tuple], coerce_na_to_none: bool) -> List[Tuple]:
        return [tuple(self._pythonize_value(v, coerce_na_to_none) for v in raw) for raw in raw_rows]

    def _divide_and_conquer_fallback(
        self,
        cursor,
        connection,
        insert_sql: str,
        pending_with_idx: List[Tuple[int, Tuple]],
        global_start_index: int,
        bad_rows_info: List[Dict[str, Any]],
    ) -> List[Tuple[int, Tuple]]:

        def salvar_chunk(chunk: List[Tuple[int, Tuple]]):
            if not chunk:
                return

            if len(chunk) == 1:
                (orig_idx, row_tuple) = chunk[0]
                try:
                    cursor.execute(insert_sql, row_tuple)
                    connection.commit()
                except Error as e_one:
                    try:
                        connection.rollback()
                    except Exception:
                        pass
                    bad_rows_info.append(
                        {"row_index": global_start_index + orig_idx, "row_data": row_tuple, "error": str(e_one)}
                    )
                return

            try:
                cursor.executemany(insert_sql, [r for (_, r) in chunk])
                connection.commit()
            except Error:
                try:
                    connection.rollback()
                except Exception:
                    pass
                mid = len(chunk) // 2
                salvar_chunk(chunk[:mid])
                salvar_chunk(chunk[mid:])

        salvar_chunk(pending_with_idx)
        return []

    def _rescue_batch_guided_by_error(
        self,
        cursor,
        connection,
        insert_sql: str,
        batch: List[Tuple],
        global_start_index: int,
    ) -> Dict[str, Any]:

        good_rows_inserted = 0
        bad_rows_info: List[Dict[str, Any]] = []
        pending = list(enumerate(batch))

        while pending:
            try:
                cursor.executemany(insert_sql, [row for (_, row) in pending])
                connection.commit()
                good_rows_inserted += len(pending)
                pending = []
                break

            except Error as e_batch:
                try:
                    connection.rollback()
                except Exception:
                    pass

                m = re.search(r"at row (\d+)", str(e_batch))
                if not m:
                    pending = self._divide_and_conquer_fallback(
                        cursor=cursor,
                        connection=connection,
                        insert_sql=insert_sql,
                        pending_with_idx=pending,
                        global_start_index=global_start_index,
                        bad_rows_info=bad_rows_info,
                    )
                    break

                bad_pos_1based = int(m.group(1))
                bad_idx_in_pending = bad_pos_1based - 1

                if not (0 <= bad_idx_in_pending < len(pending)):
                    pending = self._divide_and_conquer_fallback(
                        cursor=cursor,
                        connection=connection,
                        insert_sql=insert_sql,
                        pending_with_idx=pending,
                        global_start_index=global_start_index,
                        bad_rows_info=bad_rows_info,
                    )
                    break

                local_idx, bad_row_tuple = pending[bad_idx_in_pending]
                bad_rows_info.append(
                    {"row_index": global_start_index + local_idx, "row_data": bad_row_tuple, "error": str(e_batch)}
                )
                pending = pending[:bad_idx_in_pending] + pending[bad_idx_in_pending + 1 :]

        return {"good_rows_inserted": good_rows_inserted, "bad_rows_info": bad_rows_info}

    def _run_batches(
        self,
        *,
        cursor,
        connection,
        insert_sql: str,
        batch_iterable: Iterable[List[Tuple]],
        progress_cb: Optional[Callable[[int], None]],
        coerce_na_to_none: bool,
        ui_notify: Callable[[str], None],
        ui_skip_report: Callable[[dict], None],
    ) -> Dict[str, Any]:

        inserted_total = 0
        failed_total = 0
        batches_ok = 0
        batches_failed = 0
        bad_rows_global: List[Dict[str, Any]] = []
        running_row_start = 0

        for raw_batch in batch_iterable:
            prepared_batch = self._prepare_batch_rows(raw_batch, coerce_na_to_none)
            batch_size_here = len(prepared_batch)
            batch_start_idx = running_row_start
            batch_end_idx = running_row_start + batch_size_here - 1

            try:
                cursor.executemany(insert_sql, prepared_batch)
                connection.commit()
                inserted_total += batch_size_here
                batches_ok += 1

            except Error as err_batch:
                batches_failed += 1
                try:
                    connection.rollback()
                except Exception:
                    pass

                rescue = self._rescue_batch_guided_by_error(
                    cursor=cursor,
                    connection=connection,
                    insert_sql=insert_sql,
                    batch=prepared_batch,
                    global_start_index=batch_start_idx,
                )

                good_n = rescue["good_rows_inserted"]
                bad_list = rescue["bad_rows_info"]

                inserted_total += good_n
                failed_total += len(bad_list)
                bad_rows_global.extend(bad_list)

                ui_skip_report(
                    {
                        "range": (batch_start_idx, batch_end_idx),
                        "bad_count": len(bad_list),
                        "example_error": (bad_list[0]["error"] if bad_list else str(err_batch)),
                        "bad_rows": bad_list,
                    }
                )

            if progress_cb:
                try:
                    progress_cb(batch_size_here)
                except Exception:
                    pass

            running_row_start += batch_size_here

        return {
            "inserted": inserted_total,
            "failed": failed_total,
            "batches_ok": batches_ok,
            "batches_failed": batches_failed,
            "bad_rows": bad_rows_global,
        }

    def bulk_insert_df(
        self,
        *,
        table_name: str,
        df: pd.DataFrame,
        batch_size: int = 1000,
        use_unsafe_optimizations: bool = False,
    ) -> Dict[str, Any]:

        columns = list(df.columns)
        insert_sql = self._build_insert_sql(table_name, columns)

        cnx = self._get_connection()
        cur = cnx.cursor()

        total_rows = len(df)
        pbar = None

        skipped_total = 0
        skip_events: List[Dict[str, Any]] = []

        try:
            log_file = open("mysql_bulk_loader_last_run.log", "w", encoding="utf-8")
        except Exception:
            log_file = None

        def log_append(line: str) -> None:
            if log_file is None:
                return
            try:
                log_file.write(line.rstrip("\n") + "\n")
                log_file.flush()
            except Exception:
                pass

        def ui_notify(msg: str):
            pass

        def ui_skip_report(ev: Dict[str, Any]):
            nonlocal skipped_total, skip_events
            skipped_total += ev["bad_count"]
            skip_events.append(ev)

            batch_start, batch_end = ev["range"]
            bad_count = ev["bad_count"]
            example_err = ev["example_error"]
            bad_rows_list = ev.get("bad_rows", [])

            if pbar is not None:
                pbar.set_postfix(skipped=skipped_total, failed_batches=len(skip_events), refresh=True)

            log_append(
                f"[SKIP-BATCH] {batch_start}-{batch_end} skipped_in_batch={bad_count} "
                f"total_skipped_global={skipped_total} example_error={example_err}"
            )

            for bad in bad_rows_list:
                log_append(
                    f"[SKIP-ROW] row_index={bad.get('row_index')} batch_range={batch_start}-{batch_end} "
                    f"error={bad.get('error')} row_data={bad.get('row_data')}"
                )

        stats = None

        try:
            self._set_optimizations(cur, use_unsafe_optimizations)

            pbar = tqdm(total=total_rows, desc=f"Bulk insert DF -> {table_name}", unit="rows")

            def advance_progress(n: int):
                try:
                    pbar.update(n)
                except Exception:
                    pass

            def batch_generator_df():
                batch_rows: List[Tuple] = []
                for row in df.itertuples(index=False, name=None):
                    batch_rows.append(row)
                    if len(batch_rows) >= batch_size:
                        yield batch_rows
                        batch_rows = []
                if batch_rows:
                    yield batch_rows

            stats = self._run_batches(
                cursor=cur,
                connection=cnx,
                insert_sql=insert_sql,
                batch_iterable=batch_generator_df(),
                progress_cb=advance_progress,
                coerce_na_to_none=True,
                ui_notify=ui_notify,
                ui_skip_report=ui_skip_report,
            )

        finally:
            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass

            self._set_optimizations(cur, use_unsafe_optimizations)

            try:
                cur.close()
            except Exception:
                pass
            try:
                cnx.close()
            except Exception:
                pass

        inserted = stats["inserted"]
        failed = stats["failed"]
        total = inserted + failed if (inserted + failed) > 0 else 1
        success_pct = 100.0 * inserted / total

        print("\n=== CARGA TERMINADA ===")
        print(f"Filas procesadas totales      : {total}")
        print(f"Filas insertadas / upsert OK : {inserted}")
        print(f"Filas descartadas            : {failed}")
        print(f"Tasa de éxito                : {success_pct:.4f}%")
        print(f"Batches OK / con error       : {stats['batches_ok']} / {stats['batches_failed']}")

        return stats


def my_default_bulk_loader() -> MySQLBulkLoader:
    return MySQLBulkLoader(host=HOST, user=USER, password=PASSWORD, database=DATABASE)


def _is_read_query(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False

    # soporte simple para CTE
    q_lower = q.lstrip().lower()
    first = q_lower.split(None, 1)[0] if q_lower.split(None, 1) else ""

    return first in {"select", "with", "show", "describe", "explain"}


def execute_mysql_query(
    query: str,
    params: Optional[Tuple[Any, ...]] = None,
    *,
    host: str = HOST,
    user: str = USER,
    password: str = PASSWORD,
    database: str = DATABASE,
    fetch: bool = True,
    many: bool = False,
    raise_on_error: bool = False,  # <-- NUEVO: retrocompatible (por defecto False)
) -> Optional[pd.DataFrame]:
    """
    Ejecuta una consulta SQL sobre MySQL.

    - Queries de lectura (SELECT/WITH/SHOW/DESCRIBE/EXPLAIN): si fetch=True -> retorna DataFrame.
    - Queries de escritura (INSERT/UPDATE/DELETE/DDL/CALL con efectos): commitea automáticamente y retorna None.

    NOTA: Antes solo commiteaba cuando fetch=False. Esto generaba DML sin persistencia
          cuando se llamaba con fetch=True (default). Ahora es seguro.
    """
    cnx = None
    cur = None

    try:
        cnx = mysql.connector.connect(host=host, user=user, password=password, database=database)
        cur = cnx.cursor()

        if many and isinstance(params, list):
            cur.executemany(query, params)
        else:
            cur.execute(query, params or ())

        is_read = _is_read_query(query)

        # 1) Lectura: retornamos DF si corresponde
        if is_read and fetch:
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description] if cur.description else []
            return pd.DataFrame(rows, columns=cols)

        # 2) Escritura (o lectura con fetch=False): commit explícito
        #    También aplica para CALL que haga DML internamente.
        cnx.commit()
        return None

    except Error as e:
        # rollback para transacciones incompletas
        try:
            if cnx:
                cnx.rollback()
        except Exception:
            pass

        msg = f"[ERROR] MySQL -> {e}"
        print(msg)

        if raise_on_error:
            raise

        return None

    finally:
        if cur:
            try:
                cur.close()
            except Exception:
                pass
        if cnx:
            try:
                cnx.close()
            except Exception:
                pass
