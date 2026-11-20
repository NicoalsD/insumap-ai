# Insumap - IA

Servicio de IA sin estado de Insumap, protegido con token de servicio. Ofrece pronóstico de glucosa, triage y riesgo, detección de patrones, sugerencias de titulación, resúmenes previos a la consulta y asistente educativo con guardrails.

La IA solo apoya al médico: nunca da dosis al paciente y cada salida tiene un respaldo determinista.

## Estado

Fase 1 en servicio: el contrato v1 con el backend ([Contrato-IA](https://github.com/NicoalsD/insumap-backend/blob/develop/docs/Contrato-IA.md)).

| Endpoint | Caso | Modelo de esta versión | Respaldo |
|---|---|---|---|
| `POST /v1/glucose-forecasts` | IA1 | Tendencia amortiguada con cuantiles (`damped-trend`) | Extrapolación lineal (`fallback-linear@1.0.0`) |
| `POST /v1/risk-assessments` | IA2, IA7 | Logístico con pesos fijados a mano, sin calibrar (`risk-logistic`) | Reglas del backend |
| `POST /v1/clinical-summaries` | IA5 | LLM compatible con OpenAI (Gemini) con verificación numérica | Plantilla determinista |
| `POST /v1/assistant-replies` | IA6 | LLM con cadena de guardrails y material aprobado | Respuestas predefinidas |
| `GET /v1/health`, `/v1/ready`, `/v1/models` | | | |

Pendiente: Chronos-Bolt (IA1), modelos de scikit-learn entrenados con el simulador (IA2, IA7), `pattern-detections` (IA3) y `titration-suggestions` (IA4).

## Repositorios de Insumap

| Repositorio | Rol |
|---|---|
| [insumap-backend](https://github.com/NicoalsD/insumap-backend) | API, dominio, patrones, seguridad y documentación canónica |
| [insumap-database](https://github.com/NicoalsD/insumap-database) | Esquemas, migraciones, RLS, roles SQL y semillas (PostgreSQL 18) |
| [insumap-frontend](https://github.com/NicoalsD/insumap-frontend) | PWA del paciente, dashboard del médico y consola del admin |
| [insumap-ai](https://github.com/NicoalsD/insumap-ai) | Servicio de IA: pronóstico, triage, patrones y LLM con guardrails |
| [insumap-simulator](https://github.com/NicoalsD/insumap-simulator) | Clínica, médicos y pacientes simulados; exporta FHIR y CSV |

## Stack

Python 3.13, FastAPI, scikit-learn, Chronos-Bolt (CPU), SDK de un LLM con API compatible con OpenAI, uv, ruff, mypy `--strict` y pytest.

## Inicio rápido

```bash
uv sync
cp .env.example .env          # pon AI_SERVICE_TOKEN (y el LLM si lo usas)
set -a; . ./.env; set +a
uv run uvicorn --factory insumap_ai.main:app --port 8001   # Swagger en /docs
uv run pytest && uv run ruff check . && uv run mypy
```

Con `AI_PROVIDER_FAMILY=offline` no hay LLM: las pruebas y la CI corren sin red, y IA5 e IA6 responden con su respaldo. Para usar Gemini:

```bash
AI_PROVIDER_FAMILY=openai_compatible
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=<modelo listado en GET {LLM_BASE_URL}models>
LLM_API_KEY=<clave de Google AI Studio>
```

En el backend, `AI_SERVICE_URL=http://localhost:8001` y el mismo `AI_SERVICE_TOKEN`.

## Documentación

- [`AGENTS.md`](AGENTS.md): reglas para agentes y convenciones del repositorio.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): cómo contribuir (cuentas, ramas, commits y PR).
- [Índice de documentación canónica](https://github.com/NicoalsD/insumap-backend/tree/develop/docs).

## Equipo

- Nicolas Diaz ([`NicoalsD`](https://github.com/NicoalsD))
- Drako Salazar ([`Drako2305`](https://github.com/Drako2305))
- Nicolas Mora ([`nicomora70`](https://github.com/nicomora70))

> Proyecto académico. No es un dispositivo médico ni reemplaza el criterio del médico tratante. Los datos de pacientes son simulados.
