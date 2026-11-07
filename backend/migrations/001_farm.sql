-- FarmScan on the platform database: every farm belongs to an organization (from GeoEstate ID).
CREATE TABLE boundaries (
  id uuid PRIMARY KEY,
  org_id uuid NOT NULL,
  name text NOT NULL,
  geojson jsonb NOT NULL,
  geom geometry(Geometry, 4326) GENERATED ALWAYS AS (ST_SetSRID(ST_GeomFromGeoJSON(geojson::text), 4326)) STORED,
  area_ha double precision NOT NULL,
  created_by text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX boundaries_org ON boundaries (org_id, created_at DESC);
CREATE INDEX boundaries_geom ON boundaries USING gist (geom);

CREATE TABLE plots (
  id uuid PRIMARY KEY,
  boundary_id uuid NOT NULL REFERENCES boundaries ON DELETE CASCADE,
  plot_id integer NOT NULL,
  row_idx integer,
  col_idx integer,
  geojson jsonb NOT NULL,
  area_m2 double precision,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (boundary_id, plot_id)
);

CREATE TABLE observations (
  id bigserial PRIMARY KEY,
  boundary_id uuid NOT NULL REFERENCES boundaries ON DELETE CASCADE,
  index_name text NOT NULL,
  obs_date text NOT NULL,
  source text NOT NULL,
  scene_id text,
  cloud_cover double precision,
  valid_fraction double precision,
  mean double precision, median double precision, std double precision, p10 double precision, p90 double precision,
  pixel_count integer,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (boundary_id, index_name, obs_date, source)
);

CREATE TABLE annotations (
  id uuid PRIMARY KEY,
  boundary_id uuid NOT NULL REFERENCES boundaries ON DELETE CASCADE,
  lat double precision NOT NULL,
  lng double precision NOT NULL,
  comment text NOT NULL,
  author text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX annotations_boundary ON annotations (boundary_id, created_at DESC);

CREATE TABLE share_links (
  token text PRIMARY KEY,
  boundary_id uuid NOT NULL REFERENCES boundaries ON DELETE CASCADE,
  index_name text,
  created_by text,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Uploaded drone orthomosaics: the file is in the bucket (or backend/uploads in development)
CREATE TABLE drone_files (
  id uuid PRIMARY KEY,
  org_id uuid NOT NULL,
  filename text NOT NULL,
  storage_key text NOT NULL,
  size_bytes bigint,
  uploaded_by text,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Weather API answers, shared by all organizations (they describe places, not farms)
CREATE TABLE api_cache (
  key text PRIMARY KEY,
  payload jsonb NOT NULL,
  fetched_at double precision NOT NULL
);
