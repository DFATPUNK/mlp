import type { Pipe } from "../types/pipe";

export const mockPipes: Pipe[] = [
  {
    id: "pipe_waste_classifier",
    name: "Waste Image Classifier",
    description:
      "Classifies waste images into aluminium, cardboard, glass, biodegradable waste, or other trash.",
    type: "image_classification",
    status: "published",
    isTemplate: true,
    createdAt: "2026-05-18T00:00:00.000Z",
    updatedAt: "2026-05-18T00:00:00.000Z",
  },
];