(define (domain AUV)

(:requirements :typing :fluents)

(:types car region loc_x loc_y - object)

(:predicates(sample_taken ?r - region)
			(can_move ?c - car)
			(at_x ?x - loc_x)
			(at_y ?y - loc_y)
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

 (:action move_parallel
 :parameters (?c - car ?x1 ?x2 - loc_x ?y1 - loc_y)
 :precondition (and (can_move ?c)
 					(at_x ?x1)
 					(at_y ?y1)
 					(link_x ?x1 ?x2)
 					(not (obstacle ?x2 ?y1))
 					)
 :effect (and (not (at_x ?x1))
 			  (at_x ?x2)
 			  (increase (total_time) 1)
 			  (increase (distance) 10)
 		 )
 )

 (:action move_vertical
 :parameters (?c - car ?x1 - loc_x ?y1 ?y2 - loc_y)
 :precondition (and (can_move ?c)
 					(at_x ?x1)
 					(at_y ?y1)
 					(link_y ?y1 ?y2)
					(not (obstacle ?x1 ?y2))
 					)

 :effect (and (not (at_y ?y1))
 			  (at_y ?y2)
 			  (increase (total_time) 1)
 			  (increase (distance) 10)
 		 )
 )

 (:action move_diagonal
 :parameters (?c - car ?x1 ?x2 - loc_x ?y1 ?y2 - loc_y)
 :precondition (and (can_move ?c)
					(at_x ?x1)
					(at_y ?y1)
					(diagonallink ?x1 ?x2 ?y1 ?y2)
 					(not (obstacle ?x2 ?y1))
 					)

 :effect (and (not (at_x ?x1))
 			  (at_x ?x2)
 			  (not (at_y ?y1))
 			  (at_y ?y2)
 			  (increase (total_time) 1)
 			  (increase (distance) 14.1421356)
 		 ))

 (:action take_sample
 :parameters (?c - car ?x - loc_x ?y - loc_y ?r - region)
 :precondition (and (can_move ?c)
 					(at_x ?x)
 					(at_y ?y)
					(in_x ?x ?r)
					(in_y ?y ?r)
 					)
 :effect (and (sample_taken ?r))
 )
 )
