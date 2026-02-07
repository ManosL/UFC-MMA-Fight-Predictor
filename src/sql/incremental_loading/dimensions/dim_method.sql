MERGE INTO "DIM_Method" AS dim_method
USING (
    SELECT
        DISTINCT "Method"
    FROM
        raw_fight_stats
) AS raw_method
ON
    dim_method."DIM_Method_Name" = raw_method."Method"
-- Do nothing when match because it's pretty unlikely the fight stats to change
WHEN NOT MATCHED THEN
    INSERT (
        "DIM_Method_Name"
    )
    VALUES (
       raw_method."Method"
);
