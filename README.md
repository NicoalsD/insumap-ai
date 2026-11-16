# Insumap - IA

Servicio de IA sin estado de Insumap, protegido con token de servicio. Ofrece pronóstico de glucosa, triage y riesgo, detección de patrones, sugerencias de titulación, resúmenes previos a la consulta y asistente educativo con guardrails.

La IA solo apoya al médico: nunca da dosis al paciente y cada salida tiene un respaldo determinista.

## Estado

En construcción (Fase 0: documentación y planeación).

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

> Disponible desde la Fase 1.

```bash
uv sync
uv run uvicorn insumap_ai.main:app --reload     # servicio en http://localhost:8001
uv run pytest
uv run ruff check .
uv run mypy .
```

El modo offline (determinista) permite correr pruebas y CI sin proveedor de LLM.

## Documentación

- [`AGENTS.md`](AGENTS.md): reglas para agentes y convenciones del repositorio.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): cómo contribuir (cuentas, ramas, commits y PR).
- [Índice de documentación canónica](https://github.com/NicoalsD/insumap-backend/tree/develop/docs).

## Equipo

- Nicolas Diaz ([`NicoalsD`](https://github.com/NicoalsD))
- Drako Salazar ([`Drako2305`](https://github.com/Drako2305))
- Nicolas Mora ([`nicomora70`](https://github.com/nicomora70))

> Proyecto académico. No es un dispositivo médico ni reemplaza el criterio del médico tratante. Los datos de pacientes son simulados.
