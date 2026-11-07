-- Many farms per organization: farmer details, farms brought in from a Geohub "Farm enumeration" form, and scouting
-- tasks sent to Geohub (a "Farm scouting" form) with the scouts' findings read back.
ALTER TABLE boundaries
  ADD COLUMN details jsonb NOT NULL DEFAULT '{}',
  ADD COLUMN source text NOT NULL DEFAULT 'drawn',             -- drawn | uploaded | geohub
  ADD COLUMN source_feature uuid,                              -- the Geohub record it came from
  ADD COLUMN source_form uuid;
CREATE UNIQUE INDEX boundaries_source_feature ON boundaries (org_id, source_feature) WHERE source_feature IS NOT NULL;

CREATE TABLE org_settings (
  org_id uuid PRIMARY KEY,
  scouting_form uuid,                 -- the Geohub form (from the farm_scouting template) scouts answer in
  scouting_project uuid,
  scouting_form_name text,
  updated_by text,
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE scouting_tasks (
  id uuid PRIMARY KEY,
  boundary_id uuid NOT NULL REFERENCES boundaries ON DELETE CASCADE,
  ref text NOT NULL UNIQUE,           -- what the scout's record carries back (task_ref)
  rank integer NOT NULL,
  lat double precision NOT NULL,
  lng double precision NOT NULL,
  area_m2 double precision,
  index_name text NOT NULL,
  obs_date text,
  seen text NOT NULL,
  status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'found', 'unreachable', 'cancelled')),
  link text,
  geohub_feature uuid,
  findings jsonb,
  found_at timestamptz,
  created_by text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX scouting_tasks_boundary ON scouting_tasks (boundary_id, status, rank);
