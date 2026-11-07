-- Plots, per-plot statistics, drone orthomosaics and plant counts are gone: they need drone-grade imagery (2–5 cm),
-- and FarmScan works from 10 m satellite images. Management zones stay (they come from the satellite index).
DROP TABLE IF EXISTS plots;
DROP TABLE IF EXISTS drone_files;
