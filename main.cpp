//
//  main.cpp
//

#include "Matrix.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <mpi.h>

using namespace std;

#define Isend Send

#define PARALLEL 0
#define VERBOSE 0

#define TAG_REDUCE_WITH 4200
#define TAG_REDUCE_COL  4201
#define TAG_UPDATE_ROW 4210

// Inverser la matrice par la méthode de Gauss-Jordan; implantation séquentielle.
void invertSequential(Matrix& iA) {
    // vérifier que la matrice est carrée
    assert(iA.rows() == iA.cols());
    // construire la matrice [A I]
    MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

    // traiter chaque rangée
    for (size_t k=0; k<iA.rows(); ++k) {
        // trouver l'index p du plus grand pivot de la colonne k en valeur absolue
        // (pour une meilleure stabilité numérique).
        size_t p = k;
        double lMax = fabs(lAI(k,k));
        for(size_t i = k; i < lAI.rows(); ++i) {
            if(fabs(lAI(i,k)) > lMax) {
                lMax = fabs(lAI(i,k));
                p = i;
            }
        }
        // vérifier que la matrice n'est pas singulière
        if (lAI(p, k) == 0) throw runtime_error("Matrix not invertible");

        // échanger la ligne courante avec celle du pivot
        if (p != k) lAI.swapRows(p, k);

        double lValue = lAI(k, k);
        for (size_t j=0; j<lAI.cols(); ++j) {
            // On divise les éléments de la rangée k
            // par la valeur du pivot.
            // Ainsi, lAI(k,k) deviendra égal à 1.
            lAI(k, j) /= lValue;
        }

        // Pour chaque rangée...
        for (size_t i=0; i<lAI.rows(); ++i) {
            if (i != k) { // ...différente de k
                // On soustrait la rangée k
                // multipliée par l'élément k de la rangée courante
                double lValue = lAI(i, k);
                lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
            }
        }
    }

    // On copie la partie droite de la matrice AI ainsi transformée
    // dans la matrice courante (this).
    for (unsigned int i=0; i<iA.rows(); ++i) {
        iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols()+iA.cols(), iA.cols(), 1)];
    }
}

static void reduce(
	double const *const pivot, double *const dst,
	 size_t const col, size_t const row_width
) {
	double const val = dst[col];
	for (size_t j = 0; j < row_width; ++j) {
		dst[j] -= pivot[j] * val;
	}
}

static void reduction_process(
	std::vector<double*> const &rows, size_t const n,
	int const process_rank, int const process_count,
	size_t *const max_indices = nullptr
) {
	size_t const row_width = 2 * n;
	double *const pivot_buf = new double[row_width];

	int const next = (process_rank + 1) % process_count;
	int const previous = (process_rank + process_count - 1) % process_count;

	for (size_t k = 0; k < n; ++k) {
		double *pivot;
		size_t col;
		if (k % process_count == process_rank) {
			pivot = rows[k / process_count];
			// Si la ligne appartient au processus, faire le travail initial de
			// réduction.

			// trouver l'index du plus grand pivot de la ligne en valeur absolue
	        // (pour une meilleure stabilité numérique).
			col = 0;
	        double max = fabs(pivot[0]);
	        for (size_t j = 1; j < n; ++j) {
	            if (fabs(pivot[j]) > max) {
	                max = fabs(pivot[j]);
	                col = j;
	            }
	        }
			
	        // vérifier que la matrice n'est pas singulière
	        if (pivot[col] == 0) throw runtime_error("Matrix not invertible");
	
			// On force la valeur du pivot à valoir 1
			max = pivot[col];
			for (size_t j = 0; j < row_width; ++j) {
				pivot[j] /= max;
			}
		}
		else {
			pivot = pivot_buf;
			// Sinon, recevoir la ligne du processus précédent !
			// Obtenir la ligne avec laquelle réduire
#if VERBOSE
			cout << "Process " << process_rank << " attend ligne " << k << " du process " << previous << endl;
#endif
			MPI::COMM_WORLD.Recv(
				pivot, row_width, MPI::DOUBLE, previous, TAG_REDUCE_WITH
			);
			MPI::COMM_WORLD.Recv(&col, 1, MPI::LONG, previous, TAG_REDUCE_COL);
#if VERBOSE
			cout << "Process " << process_rank << " a recu ligne " << k << " du process " << previous << endl;
#endif
		}

		if (max_indices) {
			max_indices[k] = col;
		}

		// Envoyer la ligne pivot au prochain processus si nécessaire
		if (k % process_count != next) {
			// Envoyer au prochain process la ligne avec laquelle réduire
#if VERBOSE
			cout << "Process " << process_rank << " envoit ligne " << k << " au process " << next << endl;
#endif
			MPI::COMM_WORLD.Isend(
				pivot, row_width, MPI::DOUBLE, next, TAG_REDUCE_WITH
			);
			MPI::COMM_WORLD.Isend(&col, 1, MPI::LONG, next, TAG_REDUCE_COL);
		}

		// Appliquer la réduction sur les lignes dont on est le propriétaire
		for (size_t i = 0; i < rows.size(); ++i) {
			if (rows[i] == pivot)
				continue;
			reduce(pivot, rows[i], col, row_width);
		}
	}
}

static void invertParallelMain(Matrix &A, int const process_count) {
	size_t const n = A.cols();
    // construire la matrice [A I]
    MatrixConcatCols AI(A, MatrixIdentity(n));
	std::vector<double*> rows(
		n / process_count + (n % process_count > 0)
	);
	std::vector<size_t> max_indices(n);

	// Propagation initiale de la matrice et assignation des lignes
	for (size_t i = 0; i < n; ++i) {
		size_t const r = i % process_count;
		if (r) {
			MPI::COMM_WORLD.Isend(
				&AI(i, 0), n, MPI::DOUBLE, r, TAG_UPDATE_ROW
			);
		}
		else {
			rows[i / process_count] = &AI(i, 0);
		}
	}

	// Réduction distribuée avec Gauss-Jordan
	reduction_process(
		rows, n, 0, process_count, &max_indices[0]
	);

	// Rapatriement des lignes sur le processus
	for (size_t i = 1; i < n; ++i) {
		size_t const r = i % process_count;
		if (r) {
			MPI::COMM_WORLD.Recv(
				&AI(i, n), n, MPI::DOUBLE, r, TAG_UPDATE_ROW
			);
		}
	}
	
	// Copie du résultat dans la matrice recue en entrée
	for (unsigned int i = 0; i < n; ++i) {
        A.getRowSlice(i) = AI.getDataArray()[slice(i * AI.cols() + n, n, 1)];
	}

	// Réorganisation des lignes en temps linéaire
	size_t j = 0;
	while (j < n) {
		if (max_indices[j] != j) {
			size_t const tmp = max_indices[max_indices[j]];
			max_indices[max_indices[j]] = max_indices[j];
			A.swapRows(j, max_indices[j]);
			max_indices[j] = tmp;
		}
		else {
			++j;
		}	
	}
}

static void invertParallelSub(Matrix &A, int const process_rank, int const process_count) {
	size_t const n = A.cols() / 2;
	std::vector<double*> rows(A.rows());

	// Obtention initiale des lignes
	for (size_t i = 0; i < rows.size(); ++i) {
		size_t const j = process_rank + i * process_count;
		rows[i] = &A(i, 0);
		MPI::COMM_WORLD.Recv(
			rows[i], n, MPI::DOUBLE, 0, TAG_UPDATE_ROW
		);
		rows[i][n + j] = 1;
	}

	// Réduction distribuée avec Gauss-Jordan
	reduction_process(rows, n, process_rank, process_count);

	// Envoie des lignes possédées au processus principal
	for (size_t i = 0; i < rows.size(); ++i) {
		MPI::COMM_WORLD.Isend(
			&rows[i][n], n, MPI::DOUBLE, 0, TAG_UPDATE_ROW
		); 
	}
}

// Inverser la matrice par la méthode de Gauss-Jordan; implantation parallèle
void invertParallel(Matrix& A) {
	int const process_rank = MPI::COMM_WORLD.Get_rank();
	int const process_count = MPI::COMM_WORLD.Get_size();

	if (!process_rank)
		invertParallelMain(A, process_count);
	else
		invertParallelSub(A, process_rank, process_count);
}

// Multiplier deux matrices.
Matrix multiplyMatrix(const Matrix& iMat1, const Matrix& iMat2) {
    // vérifier la compatibilité des matrices
    assert(iMat1.cols() == iMat2.rows());
    // effectuer le produit matriciel
    Matrix lRes(iMat1.rows(), iMat2.cols());
    // traiter chaque rangée
    for(size_t i=0; i < lRes.rows(); ++i) {
        // traiter chaque colonne
        for(size_t j=0; j < lRes.cols(); ++j) {
            lRes(i,j) = (iMat1.getRowCopy(i)*iMat2.getColumnCopy(j)).sum();
        }
    }
    return lRes;
}

int main(int argc, char** argv) {
    srand((argc >= 3) ? atoi(argv[2]) : time(NULL));
    unsigned int lS = 5;
    if (argc >= 2) {
        lS = atoi(argv[1]);
    }

#if PARALLEL
	MPI::Init(argc, argv);
	int const r = MPI::COMM_WORLD.Get_rank();
	int const p = MPI::COMM_WORLD.Get_size();
	
	if (r) {
		// Un peu bizarre, mais nécessaire au respect de la signature
		// demandée pour "invertParallel".
		Matrix B(lS / p + (lS % p > r), 2 * lS);
		invertParallel(B);
	}
	else {
    	MatrixRandom lA(lS, lS);
		Matrix lB(lA);
		double start = MPI::Wtime();
		invertParallel(lB);
		double duration = MPI::Wtime() - start;
		cerr << "Exécution parallele à " << p << " processes !" << endl
			 << "Taille de la matrice: " << lS << endl
			 << "Durée du traitement: " << duration << endl
			 << "Erreur: " << multiplyMatrix(lA, lB).getDataArray().sum() - lS << endl;
	}

	MPI::Finalize();
#else
    MatrixRandom lA(lS, lS);
    Matrix lB(lA);

	double start = MPI::Wtime();
	invertSequential(lB);
	double duration = MPI::Wtime() - start;
	cerr << "Exécution à un process !" << endl
		 << "Taille de la matrice: " << lS << endl
		 << "Durée du traitement: " << duration << endl
		 << "Erreur: " << multiplyMatrix(lA, lB).getDataArray().sum() - lS << endl;
#endif

    exit(EXIT_SUCCESS);
}

