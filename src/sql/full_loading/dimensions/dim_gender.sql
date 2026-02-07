INSERT INTO "DIM_Gender" (
    "DIM_Gender_Name"
)
SELECT
    DISTINCT "Gender"
FROM
    raw_fighters_current_stats;