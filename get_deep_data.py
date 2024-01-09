import os
import math
from datetime import datetime
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.environ.get("DEEP_DB_HOST"),
    "port": os.environ.get("DEEP_DB_PORT"),
    "dbname": os.environ.get("DEEP_DB_NAME"),
    "user": os.environ.get("DEEP_DB_USER"),
    "password": os.environ.get("DEEP_DB_PASSWORD"),
}

PROJECT_ID = 3939

QUERY = """
    SELECT
        pp.original_project_id,
        ll.original_lead_id,
        ee.original_entry_id,
        af.original_af_id,
        ee.excerpt,
        ee.original_af_tags,
        ee.created_at
    FROM
        core_project pp
    INNER JOIN core_lead ll ON ll.project_id = pp.id
    INNER JOIN core_entry ee ON ee.lead_id = ll.id
    INNER JOIN core_afmapping af ON pp.af_mapping_id = af.id
    WHERE pp.original_project_id = '{}'
"""


def data_format(item):
    if isinstance(item, float) and math.isnan(item):
        return item
    lst_len = len(item)
    return [[item[i], item[lst_len // 2 + i]] for i in range(lst_len // 2)]


def connect_db(config):
    connection = psycopg2.connect(**config)
    cursor = connection.cursor()
    return cursor


def main():
    try:
        cursor = connect_db(DB_CONFIG)
        cursor.execute(QUERY.format(PROJECT_ID))
        data = cursor.fetchall()

        columns = [
            "project_id",
            "lead_id",
            "entry_id",
            "af",
            "excerpt",
            "original_af_tags",
            "entry_created",
        ]
        df = pd.DataFrame(data, columns=columns)

        tags = df["original_af_tags"].tolist()
        tags_df = pd.json_normalize(tags, max_level=0)

        matrix2d = tags_df["matrix2dWidget"]
        matrix2d = matrix2d.apply(data_format)
        tags_df["matrix2dWidget"] = matrix2d

        df_final = pd.concat([df, tags_df], axis=1)
        df_final.to_csv(f"datasets/dataset-{datetime.today().isoformat()}.csv", index=False)

    except psycopg2.Error as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
