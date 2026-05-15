"""Activity log service helpers."""

from __future__ import annotations

from datetime import timedelta
import json

import pandas as pd

from app.workspace_apps.GLA import get_connection


def load_event_types() -> list[str]:
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            SELECT DISTINCT event_type
            FROM activity_log
            ORDER BY event_type
            """
        )
        return [row["event_type"] for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def load_logs_dataframe(
    *,
    enable_date_filter: bool,
    start_date,
    end_date,
    selected_event_types: list[str],
    max_events: int,
) -> pd.DataFrame:
    conn = get_connection()
    try:
        sql = """
            SELECT
                id,
                timestamp,
                event_type,
                entity_type,
                entity_id,
                summary,
                details_json
            FROM activity_log
        """
        conditions = []
        params = []

        if enable_date_filter and start_date is not None and end_date is not None:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())
            conditions.append("timestamp < ?")
            params.append((end_date + timedelta(days=1)).isoformat())

        if selected_event_types:
            placeholders = ",".join("?" for _ in selected_event_types)
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(selected_event_types)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(int(max_events))

        rows = conn.execute(sql, params).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    rendered_rows = []
    for row in rows:
        details_text = "details: {}"
        raw_details = row["details_json"]
        if raw_details:
            try:
                details_obj = json.loads(raw_details)
                pretty = json.dumps(details_obj, indent=2, ensure_ascii=False)
                details_text = "details:\n" + "\n".join("  " + line for line in pretty.splitlines())
            except Exception:
                pass

        rendered_rows.append(
            {
                "display_text": (
                    f"[{row['timestamp']}] {row['event_type']}\n"
                    f"entity: {row['entity_type'] or '-'}"
                    f"{'' if row['entity_id'] is None else f' (id={row['entity_id']})'}\n"
                    f"summary: {row['summary'] or ''}\n\n"
                    f"{details_text}"
                )
            }
        )

    return pd.DataFrame(rendered_rows)
