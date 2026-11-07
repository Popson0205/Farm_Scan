-- Building a farm's season from the satellites takes minutes (every image in a date range), so it runs as a job.
CREATE TABLE season_jobs (
  id uuid PRIMARY KEY,
  org_id uuid NOT NULL,
  boundary_id uuid NOT NULL REFERENCES boundaries ON DELETE CASCADE,
  start_date date NOT NULL,
  end_date date NOT NULL,
  optical boolean NOT NULL,
  radar boolean NOT NULL,
  status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
  progress integer NOT NULL DEFAULT 0,
  message text,
  result jsonb,
  created_by text,
  created_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz
);
CREATE INDEX season_jobs_boundary ON season_jobs (boundary_id, created_at DESC);
