/**
 * TypeScript mirrors of the backend's Pydantic schemas.
 *
 * These are hand-written rather than generated. For a project this size the
 * generator toolchain costs more than it saves, and hand-written types let the
 * comments explain *why* a field exists. If the API grows, `openapi-typescript`
 * against `/openapi.json` is the natural next step — the schema is already
 * complete enough to drive it.
 */

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface User {
  id: string;
  email: string;
  full_name: string;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
  last_login_at: string | null;
}

export interface Profile {
  id: string;
  user_id: string;
  date_of_birth: string | null;
  height_cm: number | null;
  weight_kg: number | null;
  activity_level: string;
  diagnosis_status: string;
  dietary_preference: string;
  average_cycle_length: number | null;
  average_period_length: number | null;
  primary_goal: string | null;
  allergies: string[];
  medical_conditions: string[];
  medications: string[];
  notes: string | null;
  timezone: string;
  /** Computed server-side so BMI logic lives in exactly one place. */
  age: number | null;
  bmi: number | null;
  bmi_category: string | null;
}

export interface UserWithProfile extends User {
  profile: Profile | null;
  onboarding_complete: boolean;
}

// ------------------------------------------------------------------- chat --

export type AgentName =
  | "router"
  | "health_expert"
  | "nutrition_coach"
  | "fitness_coach"
  | "mental_wellness_coach"
  | "blood_report_analyzer"
  | "food_analyzer"
  | "habit_coach"
  | "cycle_tracker_assistant";

export interface AgentInfo {
  name: AgentName;
  display_name: string;
  description: string;
  icon: string;
  colour: string;
  capabilities: string[];
  example_prompts: string[];
}

export interface Citation {
  title: string;
  source: string;
  snippet: string;
  score: number;
  category: string | null;
}

export interface ChatMessage {
  id: string;
  conversation_id: string;
  sequence: number;
  role: "user" | "assistant" | "system";
  content: string;
  agent: string | null;
  routing_confidence: number | null;
  sources: Citation[];
  tokens_used: number | null;
  latency_ms: number | null;
  model: string | null;
  created_at: string;
}

export interface RoutingDecision {
  agent: AgentName;
  confidence: number;
  reason: string;
  matched_signals: string[];
}

export interface ChatResponse {
  conversation_id: string;
  message: ChatMessage;
  routing: RoutingDecision;
  used_rag: boolean;
  /** Facts the memory layer learned from this turn, echoed so the UI can
   *  show the user what was stored — memory they cannot see is memory they
   *  cannot correct. */
  memory_updates: Record<string, unknown>;
  suggested_followups: string[];
}

export interface Conversation {
  id: string;
  title: string;
  is_pinned: boolean;
  is_archived: boolean;
  message_count: number;
  created_at: string;
  updated_at: string;
  last_message_preview: string | null;
}

export interface ConversationDetail extends Conversation {
  messages: ChatMessage[];
  summary: string | null;
}

/** One frame of the SSE stream. */
export type StreamFrame =
  | { type: "start"; data: { conversation_id: string } }
  | {
      type: "meta";
      data: {
        agent: string;
        confidence: number;
        reason: string;
        memory_updates?: Record<string, unknown>;
        safety?: string | null;
      };
    }
  | { type: "sources"; data: { sources: Citation[] } }
  | { type: "token"; content: string }
  | {
      type: "done";
      data: {
        content: string;
        agent?: string;
        sources?: Citation[];
        tokens?: number;
        model?: string;
      };
    }
  | { type: "error"; content: string };

// -------------------------------------------------------------- prediction --

export interface FeatureContribution {
  feature: string;
  display_name: string;
  value: number | string;
  shap_value: number;
  direction: "increases" | "decreases";
  explanation: string;
}

export interface RiskAssessment {
  id: string | null;
  risk_score: number;
  risk_percentage: number;
  risk_band: "low" | "moderate" | "high";
  confidence: number;
  model_name: string;
  model_version: string;
  summary: string;
  top_factors: FeatureContribution[];
  recommendations: string[];
  disclaimer: string;
  created_at: string | null;
}

export interface ModelMetadata {
  model_name: string;
  model_version: string;
  trained_at: string;
  n_training_samples: number;
  features: string[];
  metrics: Record<string, number>;
  all_model_scores: Record<string, Record<string, number>>;
}

// --------------------------------------------------------------- analytics --

export interface TrendPoint {
  label: string;
  /** Null means "not logged", not zero — the chart must render a gap. */
  value: number | null;
}

export interface TimeSeries {
  metric: string;
  unit: string;
  points: TrendPoint[];
  average: number | null;
  change_percentage: number | null;
  direction: "up" | "down" | "flat";
}

export interface MetricCard {
  key: string;
  label: string;
  value: number | null;
  unit: string;
  secondary_label: string | null;
  secondary_value: string | null;
  change_percentage: number | null;
  direction: string;
  goal: number | null;
  progress_percentage: number | null;
  icon: string;
}

export interface AIInsight {
  title: string;
  body: string;
  category: string;
  severity: "info" | "positive" | "attention";
  action_label: string | null;
  action_url: string | null;
}

export interface Dashboard {
  generated_at: string;
  greeting: string;
  metrics: MetricCard[];
  weight_trend: TimeSeries;
  sleep_trend: TimeSeries;
  calorie_trend: TimeSeries;
  water_trend: TimeSeries;
  mood_trend: TimeSeries;
  workout_minutes_trend: TimeSeries;
  macro_split: Record<string, number>;
  habit_completion: HabitSummary[];
  cycle_summary: {
    average_length: number | null;
    regularity: string;
    days_until_next: number | null;
    predicted_next_start: string | null;
    total_cycles: number;
    variability: number | null;
  };
  latest_risk: {
    risk_score: number;
    risk_percentage: number;
    risk_band: string;
    confidence: number;
    assessed_at: string;
  } | null;
  insights: AIInsight[];
  streaks: Record<string, number>;
}

export interface HabitSummary {
  name: string;
  colour: string;
  icon: string;
  current_streak: number;
  completion_rate_30d: number;
  completed_today: boolean;
}

// ---------------------------------------------------------------- tracking --

export interface Habit {
  id: string;
  name: string;
  description: string | null;
  frequency: string;
  target_per_period: number;
  icon: string;
  colour: string;
  is_archived: boolean;
  created_at: string;
  current_streak: number;
  longest_streak: number;
  completed_today: boolean;
  completion_rate_30d: number;
}

export interface NutritionSummary {
  logged_on: string;
  total_calories: number;
  total_protein_g: number;
  total_carbs_g: number;
  total_fat_g: number;
  total_fibre_g: number;
  meal_count: number;
  calorie_goal: number;
  protein_goal_g: number;
  carb_percentage: number;
  average_glycemic_index: number | null;
}

export interface CycleInsights {
  total_cycles: number;
  average_cycle_length: number | null;
  shortest_cycle: number | null;
  longest_cycle: number | null;
  cycle_length_std_dev: number | null;
  irregular_cycle_count: number;
  regularity_label: string;
  predicted_next_start: string | null;
  predicted_fertile_window: [string, string] | null;
  days_until_next: number | null;
  average_period_length: number | null;
}

export interface Notification {
  id: string;
  type: string;
  title: string;
  body: string;
  action_url: string | null;
  icon: string;
  is_read: boolean;
  read_at: string | null;
  scheduled_for: string | null;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, string>;
  request_id?: string;
}
