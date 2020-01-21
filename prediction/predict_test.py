import query_expansion as qe
import lsi


print(lsi.predict_movies(qe.expansion_magic("泣きたい", 4)))