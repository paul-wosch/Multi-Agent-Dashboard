-- auto-generated migration

ALTER TABLE runs ADD COLUMN final_is_json INTEGER DEFAULT 0;
ALTER TABLE runs ADD COLUMN final_model TEXT;