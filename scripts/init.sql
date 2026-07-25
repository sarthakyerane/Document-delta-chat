-- delta-chat · scripts/init.sql
-- MySQL schema bootstrap (also used by docker-compose init)
-- SQLAlchemy creates the tables automatically via init_db()
-- This file is for docker-compose's docker-entrypoint-initdb.d injection

CREATE DATABASE IF NOT EXISTS delta_chat CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE delta_chat;

-- Tables are created by SQLAlchemy's Base.metadata.create_all()
-- This file only ensures the database and user exist.
GRANT ALL PRIVILEGES ON delta_chat.* TO 'delta_user'@'%';
FLUSH PRIVILEGES;
