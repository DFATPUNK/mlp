import type { DatasetProvider, ProviderConnection, ProviderDataset } from "../../types/builder";

const makeRows = (count: number, rowBuilder: (index: number) => Record<string, unknown>) => Array.from({ length: count }, (_, i) => rowBuilder(i));

const datasetRegistry: Record<DatasetProvider, ProviderDataset[]> = {
  huggingface: [
    { externalId: "hf:titanic_small", name: "Titanic (small tabular)", sourceLabel: "Hugging Face • Titanic", sourceConfig: { dataset: "titanic_small", split: "train" }, rows: makeRows(120, (i) => ({ passenger_id: `P-${1000 + i}`, age: 18 + (i % 60), fare: Number((20 + i * 0.7).toFixed(2)), survived: i % 2, embark_date: `2024-01-${String((i % 28) + 1).padStart(2, "0")}` })) },
  ],
  airtable: [
    { externalId: "airtable:base_sales:leads:all", name: "Leads table (All leads)", sourceLabel: "Airtable • Sales Base / Leads / All leads", sourceConfig: { base_id: "base_sales", table_id: "leads", view_id: "all" }, rows: makeRows(160, (i) => ({ lead_id: `L-${3000 + i}`, source: ["Web", "Referral", "Ad"][i % 3], region: ["US", "EU", "APAC"][i % 3], employee_count: 5 + (i % 500), qualified: i % 4 === 0 ? "yes" : "no" })) },
  ],
  google_sheets: [
    { externalId: "sheets:revenue_forecast:q2", name: "Revenue Forecast Q2", sourceLabel: "Google Sheets • Revenue Forecast / Q2", sourceConfig: { spreadsheet_id: "spreadsheet_revenue", sheet_name: "Q2", range: "A1:H400" }, rows: makeRows(220, (i) => ({ row_id: i + 1, account_tier: ["SMB", "Mid", "Enterprise"][i % 3], monthly_active_users: 50 + i * 2, discount_pct: i % 15, expected_revenue: 1000 + i * 15, renewal_date: `2026-${String((i % 12) + 1).padStart(2, "0")}-01` })) },
  ],
};

export function getMockConnections(provider: DatasetProvider): ProviderConnection[] {
  const accountLabel = provider === "huggingface" ? "demo@huggingface" : provider === "airtable" ? "Sales workspace" : "Growth analytics";
  return [{ id: `${provider}-connection-demo`, provider, providerAccountId: `${provider}-acct-demo`, providerAccountLabel: accountLabel, scopes: ["read"], expiresAt: null }];
}

export function getMockDatasets(provider: DatasetProvider): ProviderDataset[] {
  return datasetRegistry[provider];
}
