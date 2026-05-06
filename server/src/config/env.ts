import dotenv from "dotenv";
dotenv.config();

function required(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

function optional(name: string, fallback: string): string {
  return process.env[name] ?? fallback;
}

export const env = {
  supabaseUrl: required("SUPABASE_URL"),
  supabaseServiceRoleKey: required("SUPABASE_SERVICE_ROLE_KEY"),
  fmpApiKey: required("FMP_API_KEY"),
  alpacaApiKey: required("ALPACA_API_KEY"),
  alpacaSecretKey: required("ALPACA_SECRET_KEY"),
  alpacaBaseUrl: optional("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
  port: parseInt(optional("PORT", "5002"), 10),
  nodeEnv: optional("NODE_ENV", "development"),
};
