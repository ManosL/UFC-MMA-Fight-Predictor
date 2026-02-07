-- MAYBE IT WILL BE POPULATED USING PYTHON BUT IT CAN BE DONE WITH POSTGRES

DROP TABLE IF EXISTS "DIM_Date";

CREATE TABLE IF NOT EXISTS "DIM_Date" (
    "DIM_Date_Key"                  VARCHAR(10) PRIMARY KEY,
    "DIM_Date_Full"                 VARCHAR(40) NOT NULL, -- TODO: CONVERT THIS TO DATE
    "DIM_Date_Day"                  SMALLINT NOT NULL,
    "DIM_Date_Day_Name"             VARCHAR(20) NOT NULL,
    "DIM_Date_Month"                SMALLINT NOT NULL,
    "DIM_Date_Month_Name"           VARCHAR(20) NOT NULL,
    "DIM_Date_Year"                 SMALLINT NOT NULL,
    "DIM_Date_Day_of_Year"          SMALLINT NOT NULL,
    "DIM_Date_Day_of_Week"          SMALLINT NOT NULL,
    "DIM_Date_Quarter"              SMALLINT NOT NULL,
    "DIM_Date_Is_Month_Start"       BOOLEAN NOT NULL,
    "DIM_Date_Is_Month_End"         BOOLEAN NOT NULL,
    "DIM_Date_Is_Quarter_Start"     BOOLEAN NOT NULL,
    "DIM_Date_Is_Quarter_End"       BOOLEAN NOT NULL,
    "DIM_Date_Is_Year_Start"        BOOLEAN NOT NULL,
    "DIM_Date_Is_Year_End"          BOOLEAN NOT NULL,
    "DIM_Date_Is_Leap_Year"         BOOLEAN NOT NULL
);