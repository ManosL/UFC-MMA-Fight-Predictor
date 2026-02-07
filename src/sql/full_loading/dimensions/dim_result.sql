INSERT INTO "DIM_Result" (
    "DIM_Result_Name"
)
SELECT
    DISTINCT "Result"
FROM
    raw_fight_stats;