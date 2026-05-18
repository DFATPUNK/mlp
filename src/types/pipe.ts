export type PipeStatus = "draft" | "ready" | "published" | "disabled";

export type PipeType =
  | "tabular_classification"
  | "tabular_regression"
  | "image_classification";

export type Pipe = {
  id: string;
  name: string;
  description: string;
  type: PipeType;
  status: PipeStatus;
  isTemplate: boolean;
  createdAt: string;
  updatedAt: string;
};

export type WasteClassifierOutput = {
  predicted_category:
    | "aluminium"
    | "cardboard"
    | "glass"
    | "biodegradable"
    | "other_trash";
  confidence: number;
  alternative_categories: {
    label: string;
    confidence: number;
  }[];
};