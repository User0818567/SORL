(define (domain Taxi)

(:requirements :typing :fluents)

(:types car region loc_x loc_y person - object)

(:predicates(can_move ?c - car)
			(car_at_x ?x - loc_x)
			(car_at_y ?y - loc_y)
			(p_at ?p - person ?r - region)

			(in_car ?p - person ?c - car)

			(in_x ?x - loc_x ?r - region)
			(in_y ?y - loc_y ?r - region)

			(obstacle ?x - loc_x ?y - loc_y)

			(link_x ?x1 - loc_x ?x2 - loc_x)
			(link_y ?y1 - loc_y ?y2 - loc_y)

			(diagonallink ?x1 - loc_x ?x2 - loc_x ?y1 - loc_y ?y2 - loc_y)
)

(:functions (total_time)
            (distance)
            )



 (:action take_on
 :parameters (?c - car ?x - loc_x ?y - loc_y ?r - region ?p - person)
 :precondition (and (can_move ?c)
					(not (in_car ?p ?c))
                    (p_at ?p ?r)
                    (in_x ?x ?r)
                    (in_y ?y ?r)
                    (car_at_x ?x)
                    (car_at_y ?y)
 					)
 :effect (and (in_car ?p ?c)
 			  (not (p_at ?p ?r))
 )
 )

(:action take-off
 :parameters (?c - car ?x - loc_x ?y - loc_y ?r - region ?p - person)
 :precondition (and (in_car ?p ?c)
                    (in_x ?x ?r)
                    (in_y ?y ?r)
                    (car_at_x ?x)
                    (car_at_y ?y)
                 )
 :effect (and (not (in_car ?p ?c))
              (p_at ?p ?r)
              )
 )


  (:action move_parallel
 :parameters (?c - car ?x1 ?x2 - loc_x ?y1 - loc_y)
 :precondition (and (can_move ?c)
 					(car_at_x ?x1)
 					(car_at_y ?y1)
 					(link_x ?x1 ?x2)
 					(not (obstacle ?x2 ?y1))
 					)

 :effect (and (not (car_at_x ?x1))
 			  (car_at_x ?x2)
 			  (increase (total_time) 1)
 			  (increase (distance) 20)
 		 )
 )

 (:action move_vertical
 :parameters (?c - car ?x1 - loc_x ?y1 ?y2 - loc_y)
 :precondition (and (can_move ?c)
 					(car_at_x ?x1)
 					(car_at_y ?y1)
 					(link_y ?y1 ?y2)
					(not (obstacle ?x1 ?y2))
 					)

 :effect (and (not (car_at_y ?y1))
 			  (car_at_y ?y2)
 			  (increase (total_time) 1)
 			  (increase (distance) 20)
 		 )
 )

 (:action move_diagonal
 :parameters (?c - car ?x1 ?x2 - loc_x ?y1 ?y2 - loc_y)
 :precondition (and (can_move ?c)
					(car_at_x ?x1)
					(car_at_y ?y1)
					(diagonallink ?x1 ?x2 ?y1 ?y2)
 					(not (obstacle ?x2 ?y1))
 					)

 :effect (and (not (car_at_x ?x1))
 			  (car_at_x ?x2)
 			  (not (car_at_y ?y1))
 			  (car_at_y ?y2)
 			  (increase (total_time) 1)
 			  (increase (distance) 28.2842712)
 		 ))
 )
