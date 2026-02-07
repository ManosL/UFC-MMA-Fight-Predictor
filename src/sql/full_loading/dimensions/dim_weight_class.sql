INSERT INTO "DIM_Weight_Class" (
    "DIM_Weight_Class_Name"
)
SELECT
    DISTINCT "Weight_Class"
FROM
    raw_fight_stats
ORDER BY
    "Weight_Class";