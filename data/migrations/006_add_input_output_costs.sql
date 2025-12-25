-- auto-generated migration

ALTER TABLE agent_metrics ADD COLUMN input_cost REAL;
ALTER TABLE agent_metrics ADD COLUMN output_cost REAL;