import os
import pandas as pd
import psycopg2
from pandas.core.frame import DataFrame
from sqlalchemy import create_engine, engine, inspect
from dotenv import load_dotenv
import psycopg2.extras

load_dotenv()


def write_df_to_postgres(table_name: str, df: DataFrame, if_exists: str = "replace"):
    """

    Args:
        table_name:
        df:
        if_exists: How to behave if the table already exists.
            fail: Raise a ValueError.
            replace: Drop the table before inserting new values.
            append: Insert new values to the existing table.

    Returns:

    """
    eng = get_connection(conn_type="sqlalchemy")
    with eng.connect() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)


def get_table(table_name: str) -> pd.DataFrame:
    statement = f'select * from "{table_name}"'
    df = postgres_to_df(statement)
    return df


def get_distinct_col_values(table_name: str, column_name: str):
    statement = f"select distinct {column_name} from {table_name}"
    df = postgres_to_df(statement)
    return df


def get_table_with_condition(table_name: str, column_name: str, condition: str) -> pd.DataFrame:
    statement = f"select * from {table_name} where {column_name} = '{condition}'"
    df = postgres_to_df(statement)
    return df


def get_table_with_condition_list(table_name: str, column_name: str, conditions: list) -> pd.DataFrame:
    statement = f"""select * from {table_name} where {column_name} in ('{"', '".join(conditions)}')"""
    df = postgres_to_df(statement)
    return df


def get_table_with_condition_df(table_name: str, conditions: pd.DataFrame) -> pd.DataFrame:
    """
    Gets db-table based on multiple conditions for multiple columns
    Args:
        table_name: name of table in db
        conditions: the conditions have to be a df which contain values to filter for. column_names of df have to
         respond to names of columns in db

    Returns:

    """
    and_str = " and "
    or_str = " ) or ( "
    statement = f"""select * from {table_name} where ( """

    for _, condition in conditions.iterrows():

        for column_name, value in condition.iteritems():
            statement += f""" {column_name} = '{value}' {and_str}"""
        statement = statement.rstrip().rstrip(and_str)
        statement += or_str
    statement = statement.rstrip().rstrip(or_str)
    statement += " ) "
    df = postgres_to_df(statement)
    return df


def get_table_with_condition_dict(table_name: str, conditions: dict) -> pd.DataFrame:
    """
    Gets db-table based on multiple conditions for multiple columns
    Args:
        table_name: name of table in db
        conditions: The keys have to correspond to columns in the db. The values have to be a list which correspond to
            respective values of that column.

    Returns:

    """
    and_str = " and "
    statement = f"""select * from {table_name} where """
    for column_name, condition_list in conditions.items():
        statement += f""" {column_name} in ('{"', '".join(map(str, condition_list))}') {and_str}"""
    statement = statement.rstrip().rstrip(and_str)
    df = postgres_to_df(statement)
    return df


def has_table(table_name: str) -> bool:
    conn_engine = get_connection("sqlalchemy")
    insp = inspect(conn_engine)
    return insp.has_table(table_name)


def value_exists_in_column(table_name: str, column_name: str, value: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"select distinct 1 from {table_name} where {column_name} = '{value}'")
    conn.commit()
    cur.close()
    return cur.rowcount >= 1


def postgres_to_df(stmt) -> DataFrame:
    """[summary]
        Create a dataframe from an sql statement. 
        A connection is open and closed while executing this select statement.
    Parameters
    ----------
    stmt : [str]
        [description]
        SQL statement as String
    Returns
    -------
    [DataFrame]
        [description]
        DataFrame created from the select statement.
    """
    conn = get_connection(conn_type="sqlalchemy")
    df = pd.read_sql_query(stmt, conn)
    conn.dispose()
    return df


def delete_stm(table_name: str, column_name: str, column_value_to_delete):
    """
    Delete all rows based on a single condition.
    Args:
        table_name:
        column_name:
        column_value_to_delete:

    Returns:

    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"delete from {table_name} where {column_name} = '{column_value_to_delete}'")
    conn.commit()
    cur.close()


def truncate_table(table_name: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"truncate table {table_name}")
    conn.commit()
    cur.close()


def drop_table(table_name: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f'drop table if exists "{table_name}"')
    conn.commit()
    cur.close()


def get_connection(conn_type: str = "postgres"):
    """
    Returns connection using variables of env file

    Args:
        conn_type: "postgres" or "sqlalchemy"

    Returns:

    """
    user = os.environ.get('POSTGRES_USR')
    pwd = os.environ.get('POSTGRES_PWD')
    db = os.environ.get('POSTGRES_DB')
    host = os.environ.get('POSTGRES_HOST')
    port = os.environ.get('POSTGRES_PORT')
    client_encoding = "utf8"

    if conn_type == "sqlalchemy":
        database_uri = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?client_encoding={client_encoding}"
        conn = create_engine(database_uri)
    elif conn_type == "postgres":
        conn = psycopg2.connect(database=db,
                                user=user,
                                password=pwd,
                                host=host,
                                port=port)
    else:
        raise Exception("#85 Conn_type unknown!")

    return conn


def is_table_empty(table_name: str):
    table_filled = 1
    statement = f"select count({table_filled}) where exists (select * from {table_name})"
    df = postgres_to_df(statement)
    return df["count"].iloc[0] != table_filled


if __name__ == "__main__":
    print("Get connection using env variables")
    con = get_connection()
    print("Connection", con)
    con.close()
    print("Closing conection")
