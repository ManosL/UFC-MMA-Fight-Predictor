MERGE INTO "DIM_Result" AS dim_result
USING (
    SELECT
        DISTINCT "Result"
    FROM
        raw_fight_stats
) AS raw_result
ON
    dim_result."DIM_Result_Name" = raw_result."Result"
-- Do nothing when match because it's pretty unlikely the fight stats to change
WHEN NOT MATCHED THEN
    INSERT (
        "DIM_Result_Name"
    )
    VALUES (
       raw_result."Result"
);
