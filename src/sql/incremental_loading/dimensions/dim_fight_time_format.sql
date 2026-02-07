MERGE INTO "DIM_Fight_Time_Format" AS dim_fight_time_format
USING (
    SELECT
        DISTINCT "Fight_Time_Format"
    FROM
        raw_fight_stats
) AS raw_fight_time_format
ON
    dim_fight_time_format."DIM_Fight_Time_Format_Name" = raw_fight_time_format."Fight_Time_Format"
-- Do nothing when match because it's pretty unlikely the fight stats to change
WHEN NOT MATCHED THEN
    INSERT (
        "DIM_Fight_Time_Format_Name"
    )
    VALUES (
       raw_fight_time_format."Fight_Time_Format"
    );
