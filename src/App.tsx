import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { CreatePipePage } from "./routes/CreatePipePage";
import { LoginPage } from "./routes/LoginPage";
import { PipeDetailPage } from "./routes/PipeDetailPage";
import { PipesPage } from "./routes/PipesPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/app/pipes" replace />} />
      <Route path="/login" element={<LoginPage />} />

      <Route element={<ProtectedRoute />}>
        <Route path="/app" element={<AppShell />}>
          <Route path="pipes" element={<PipesPage />} />
          <Route path="pipes/new" element={<CreatePipePage />} />
          <Route path="pipes/:pipeId" element={<PipeDetailPage />} />
        </Route>
      </Route>

      <Route path="*" element={<Navigate to="/app/pipes" replace />} />
    </Routes>
  );
}