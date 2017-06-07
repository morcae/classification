# ML classification

4 genres: blues, classical, pop, rock (might be more)
80:20 traint:test sets

* classification with full song or segment
* one genre or estimation
* configuration in cnf.conf file

## algorithms used:
* naive bayesian (54 out of 80 predicted)
* linear SVM (60)
* k nearest neighbours (51)
* Random Forest (54)
* Ridge (60)

Results of Ridge classifier are saved in database and used in web interface for experts.


# Web interface
* Bottle framework
* Test set is used for experts' classification
* Experts' results are saved in database
* Bar chart of all results for each song
