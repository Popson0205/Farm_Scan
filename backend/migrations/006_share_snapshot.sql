-- A share link keeps the result it was made from (the map overlay and its numbers), so it still opens after a
-- restart and shows what was shared, not whatever was run last.
ALTER TABLE share_links
  ADD COLUMN overlay_png bytea,
  ADD COLUMN overlay_bounds jsonb,
  ADD COLUMN stats jsonb,
  ADD COLUMN label text,
  ADD COLUMN obs_date text,
  ADD COLUMN source text;
