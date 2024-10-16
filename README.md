# FedFlow
A federated learning framework with a focus on architecture and scalability.

## Features
- [x] Centralized FL
- [x] Semi-decentralized FL
- [x] Decentralized FL
- [x] Different aggregations
- [ ] Homogeneous/Non-homogeneous data distribution
- [ ] Different models
- [ ] Different datasets

## Quickstart
Head over to `evaluation/` and run the following commands:
```bash
python3 scenario_creator.py
```
You will be asked to create your scenario, after that a `docker-compose.yml` file will be created. You can run the scenario with:
```bash
docker compose up 
```
