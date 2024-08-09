# app/database.py
import mysql.connector
from mysql.connector import Error
import logging
import urllib.parse
password = urllib.parse.quote_plus("!Q@W3e4r")  # 특수 문자를 URL 인코딩

logger = logging.getLogger(__name__)


DB_CONFIG = {
    'user': 'manager',
    'password': '!Q@W3e4r',
    'host': '211.37.179.178',
    'port': 3306,
    'database': 'spoon'
}

def create_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        
        if connection.is_connected():
            logger.info("MySQL 데이터베이스에 연결되었습니다")
            return connection
    except Error as e:
        logger.error(f"Error: '{e}'")
        return None

def close_connection(connection):
    if connection.is_connected():
        connection.close()
        logger.info("MySQL 데이터베이스 연결이 종료되었습니다")
