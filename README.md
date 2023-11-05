![Logo](./.docs/images/logo.svg)

## Description
This application is the Recommendation System for Affililab, it requires connection to MonogoDB and connection to Milvus Vector Database.

[https://www.mongodb.com/](https://www.mongodb.com/)
[https://milvus.io/](https://milvus.io/)

## 1.Install

### install requirements

```
pip install -r requirements
```

## 2. set env variables
```
DB_MONGODB_CONNECTION=
DB_MONGODB_DATABASE=
MILVUS_DATABASE_HOST=
MILVUS_DATABASE_PORT=
```
## 3. start server
```sh
python wsgi.py
```
