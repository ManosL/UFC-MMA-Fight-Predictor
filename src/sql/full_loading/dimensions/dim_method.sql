INSERT INTO "DIM_Method" (
    "DIM_Method_Name"
)
SELECT
    DISTINCT "Method"
FROM
    raw_fight_stats;