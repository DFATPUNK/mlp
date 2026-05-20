import type { DatasetProvider, ProviderConnection, ProviderDataset } from "../../types/builder";
import { listHuggingFaceDatasets } from "./huggingFaceDatasets";

const makeRows = (count: number, rowBuilder: (index: number) => Record<string, unknown>) => Array.from({ length: count }, (_, i) => rowBuilder(i));

const datasetRegistry: Record<Exclude<DatasetProvider, "huggingface">, ProviderDataset[]> = {
  airtable: [
    { externalId: "airtable:base_sales:leads:all", name: "Leads table (All leads)", sourceLabel: "Airtable (Alpha mock) • Sales Base / Leads / All leads", sourceConfig: { base_id: "base_sales", table_id: "leads", view_id: "all", mock: true }, rows: makeRows(160, (i) => ({ lead_id: `L-${3000 + i}`, source: ["Web", "Referral", "Ad"][i % 3], region: ["US", "EU", "APAC"][i % 3], employee_count: 5 + (i % 500), qualified: i % 4 === 0 ? "yes" : "no" })) },
  ],
  google_sheets: [
    { externalId: "sheets:revenue_forecast:q2", name: "Revenue Forecast Q2", sourceLabel: "Google Sheets (Alpha mock) • Revenue Forecast / Q2", sourceConfig: { spreadsheet_id: "spreadsheet_revenue", sheet_name: "Q2", range: "A1:H400", mock: true }, rows: makeRows(220, (i) => ({ row_id: i + 1, account_tier: ["SMB", "Mid", "Enterprise"][i % 3], monthly_active_users: 50 + i * 2, discount_pct: i % 15, expected_revenue: 1000 + i * 15, renewal_date: `2026-${String((i % 12) + 1).padStart(2, "0")}-01` })) },
  ],
};

export function getMockConnections(provider: DatasetProvider): ProviderConnection[] {
  const accountLabel = provider === "huggingface" ? "Hugging Face read-only" : provider === "airtable" ? "Airtable alpha mock" : "Google Sheets alpha mock";
  return [{ id: `${provider}-connection-demo`, provider, providerAccountId: `${provider}-acct-demo`, providerAccountLabel: accountLabel, scopes: ["read"], expiresAt: null }];
}

export function getProviderDatasets(provider: DatasetProvider): ProviderDataset[] {
  if (provider === "huggingface") return listHuggingFaceDatasets();
  return datasetRegistry[provider];
}
