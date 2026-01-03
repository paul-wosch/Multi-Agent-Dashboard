-- auto-generated migration

ALTER TABLE agents ADD COLUMN system_prompt_template TEXT;
ALTER TABLE agent_run_configs ADD COLUMN system_prompt_template TEXT;