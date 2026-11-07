-- Weekly farm alerts (by email, to the people who ask for them) and farm verification reports.
CREATE TABLE alert_subscribers (
  org_id uuid NOT NULL,
  user_id text NOT NULL,
  email text NOT NULL,
  name text,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (org_id, user_id)
);

CREATE TABLE alert_runs (
  id bigserial PRIMARY KEY,
  org_id uuid NOT NULL,
  sent_at timestamptz NOT NULL DEFAULT now(),
  farms integer NOT NULL,
  alerts jsonb NOT NULL,
  recipients integer NOT NULL
);
CREATE INDEX alert_runs_org ON alert_runs (org_id, sent_at DESC);

CREATE TABLE farm_reports (
  id uuid PRIMARY KEY,
  code text NOT NULL UNIQUE,                  -- printed on the report; anyone can check it at /verify.html?code=…
  org_id uuid NOT NULL,
  org_name text,
  boundary_id uuid REFERENCES boundaries ON DELETE SET NULL,
  content jsonb NOT NULL,
  created_by text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX farm_reports_org ON farm_reports (org_id, created_at DESC);

-- What the organization's licence allowed when a member last opened FarmScan: the weekly alerts go out on their own,
-- with nobody signed in, and only while the licence includes them.
ALTER TABLE org_settings
  ADD COLUMN org_name text,
  ADD COLUMN features jsonb NOT NULL DEFAULT '[]',
  ADD COLUMN licence text,
  ADD COLUMN seen_at timestamptz;
