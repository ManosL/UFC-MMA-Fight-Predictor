MERGE INTO "DIM_Gender" AS dim_gender
USING (
    SELECT
        DISTINCT "Gender"
    FROM
        raw_fighters_current_stats
) AS raw_gender
ON
    dim_gender."DIM_Gender_Name" = raw_gender."Gender"
-- Do nothing when match because it's pretty unlikely the fight stats to change
WHEN NOT MATCHED THEN
    INSERT (
        "DIM_Gender_Name"
    )
    VALUES (
       raw_gender."Gender"
);
