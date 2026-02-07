INSERT INTO "DIM_Fight_Time_Format"(
    "DIM_Fight_Time_Format_Name"
)
SELECT
    DISTINCT "Fight_Time_Format"
FROM
    raw_fight_stats
ORDER BY
    "Fight_Time_Format";