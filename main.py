import xgboost_ensemble
import neural_network_projections
import bayesian_neural_network_projections
import rf_projections
import svr_projections
import rr_projections

def main():
    neural_network_projections.make_projections(False, False, 2024, True)
    bayesian_neural_network_projections.make_projections(True, False, 2024, True)
    rf_projections.make_projections(True, False, 2024, True)
    svr_projections.make_projections(True, False, 2024, True)
    rr_projections.make_projections(True, False, 2024, True)

    xgboost_ensemble.make_projections(True, False, 2024, True)

if __name__ == '__main__':
    main()
