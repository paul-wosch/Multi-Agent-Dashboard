-- auto-generated migration

ALTER TABLE agent_outputs ADD COLUMN is_json INTEGER DEFAULT 0;
ALTER TABLE agent_outputs ADD COLUMN model TEXT;