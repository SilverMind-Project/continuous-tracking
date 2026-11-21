SET search_path = continuous_tracking, public;

-- Allow the paired image-only evaluation table to record the household appearance
-- classifier arm alongside the existing control and treatment arms.
ALTER TABLE continuous_tracking.image_only_evaluation_daily
    DROP CONSTRAINT IF EXISTS image_only_evaluation_daily_arm_check;

ALTER TABLE continuous_tracking.image_only_evaluation_daily
    ADD CONSTRAINT image_only_evaluation_daily_arm_check
    CHECK (arm IN ('control', 'treatment', 'classifier'));
