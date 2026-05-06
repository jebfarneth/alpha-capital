import express from "express";
import { env } from "./config/env";

const app = express();
app.use(express.json());

app.get("/health", (_req, res) => {
  res.json({
    status: "ok",
    app: "alpha-capital",
    env: env.nodeEnv,
    timestamp: new Date().toISOString(),
  });
});

app.listen(env.port, () => {
  console.log(`[alpha-capital] listening on :${env.port} (${env.nodeEnv})`);
});
