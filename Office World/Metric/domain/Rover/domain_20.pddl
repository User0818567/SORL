(define (domain Rover)

(:requirements :typing :fluents)

(:types rover waypoint store camera mode lander objective)

(:predicates(at_x ?x - loc_x)
			(at_y ?y - loc_y)
			(in_x ?x - loc_x ?w - waypoint)
			(in_y ?y - loc_y ?w - waypoint)

			(obstacle ?x - loc_x ?y - loc_y)

			(link_x ?x1 - loc_x ?x2 - loc_x)
			(link_y ?y1 - loc_y ?y2 - loc_y)

			(diagonallink ?x1 - loc_x ?x2 - loc_x ?y1 - loc_y ?y2 - loc_y)

			(channel_free ?l - lander)
			(at_lander ?x - lander ?y - waypoint)
	        (equipped_for_soil_analysis ?r - rover)
         	(equipped_for_rock_analysis ?r - rover)
            (equipped_for_imaging ?r - rover)
            (empty ?s - store)
            (have_rock_analysis ?r - rover ?w - waypoint)
            (have_soil_analysis ?r - rover ?w - waypoint)
            (full ?s - store)
	        (calibrated ?c - camera ?r - rover) 
	        (supports ?c - camera ?m - mode)
            (visible ?w - waypoint ?p - waypoint)
            (have_image ?r - rover ?o - objective ?m - mode)
            (communicated_soil_data ?w - waypoint)
            (communicated_rock_data ?w - waypoint)
            (communicated_image_data ?o - objective ?m - mode)
	        (at_soil_sample ?w - waypoint)
	        (at_rock_sample ?w - waypoint)
            (visible_from ?o - objective ?w - waypoint)
	        (store_of ?s - store ?r - rover)
	        (calibration_target ?i - camera ?o - objective)
	        (on_board ?i - camera ?r - rover)
	        (in_sun ?w - waypoint)


)

(:functions (total_time)
            (distance)
            )

 (:action move_parallel
 :parameters (?r - rover ?x1 - loc_x ?x2 - loc_x ?y1 - loc_y)
 :precondition (and 
 					(at_x ?x1)
 					(at_y ?y1)
 					(link_x ?x1 ?x2)
 					(not (obstacle ?x2 ?y1))
 					)
 :effect (and (not (at_x ?x1))
 			  (at_x ?x2)
 			  (increase (total_time) 1)
 			  (increase (distance) 20)
 		 )
 )

 (:action move_vertical
 :parameters (?r - rover ?x1 - loc_x ?y1 - loc_y ?y2 - loc_y)
 :precondition (and
 					(at_x ?x1)
 					(at_y ?y1)
 					(link_y ?y1 ?y2)
					(not (obstacle ?x1 ?y2))
 					)

 :effect (and (not (at_y ?y1))
 			  (at_y ?y2)
 			  (increase (total_time) 1)
 			  (increase (distance) 20)
 		 )
 )

 (:action move_diagonal
 :parameters (?r - rover ?x1 - loc_x ?x2 - loc_x ?y1 - loc_y ?y2 - loc_y)
 :precondition (and 
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
 			  (increase (distance) 28.2842712)
 		 ))

 
 (:action sample_soil
:parameters (?s - store ?w - waypoint ?r - rover ?x - loc_x ?y - loc_y)
:precondition (and (at_x ?x)
				           (at_y ?y)				
				           (in_x ?x ?w)
				           (in_y ?y ?w)
                   (at_soil_sample ?w) 
                   (equipped_for_soil_analysis ?r) 
                   (store_of ?s ?r) 
                   (empty ?s)
                   (not (full ?s))
		)
:effect (and (not (empty ?s)) 
                  (full ?s) 
                  (have_soil_analysis ?r ?w) 
                  (not (at_soil_sample ?w))
		)
)

 (:action sample_rock
:parameters (?r - rover ?s - store ?w - waypoint ?x - loc_x ?y - loc_y)
:precondition (and  (at_x ?x)
				    (at_y ?y)				
				    (in_x ?x ?w)
				    (in_y ?y ?w)
                    (at_rock_sample ?w)
                    (equipped_for_rock_analysis ?r)
                    (store_of ?s ?r)
                    (empty ?s)
                    (not (full ?s))
		)
:effect (and (not (empty ?s))
             (full ?s)
             (have_rock_analysis ?r ?w)
             (not (at_rock_sample ?w))
		)
)

(:action drop
:parameters (?x - rover ?y - store)
:precondition (and (store_of ?y ?x)
                   (full ?y)
		)
:effect (and (not (full ?y))
             (empty ?y)
	)
)


(:action calibrate
 :parameters (?r - rover ?i - camera ?t - objective ?w - waypoint ?x - loc_x ?y - loc_y)
 :precondition (and (equipped_for_imaging ?r)
                    (visible_from ?t ?w)
                    (on_board ?i ?r)
                    (at_x ?x)
				    (at_y ?y)				
				    (in_x ?x ?w)
				    (in_y ?y ?w)
		)
 :effect (and (calibrated ?i ?r) )
)

(:action take_image
 :parameters (?r - rover ?w - waypoint ?o - objective ?i - camera ?m - mode ?x - loc_x ?y - loc_y)
 :precondition (and (calibrated ?i ?r)
             		(on_board ?i ?r)
                    (equipped_for_imaging ?r)
                    (supports ?i ?m)
			        (visible_from ?o ?w)
                    (at_x ?x)
				    (at_y ?y)				
				    (in_x ?x ?w)
				    (in_y ?y ?w)
               )
 :effect (and (have_image ?r ?o ?m)
              (not (calibrated ?i ?r))
		)
)

(:action communicate_rock_data
 :parameters (?r - rover ?l - lander ?p - waypoint ?w - waypoint ?y - waypoint ?x - loc_x ?yy - loc_y)
 :precondition (and (at_x ?x)
				    (at_y ?yy)				
				    (in_x ?x ?w)
				    (in_y ?yy ?w)
                    (at_lander ?l ?y)
                    (have_rock_analysis ?r ?p)
                    (visible ?w ?y)
            )
 :effect (and   
                (communicated_rock_data ?p)
          )
)

(:action communicate_soil_data
 :parameters (?r - rover ?l - lander ?p - waypoint ?w - waypoint ?y - waypoint ?xx - loc_x ?yy - loc_y)
 :precondition (and 
                    (at_x ?xx)
                    (at_y ?yy)        
                    (in_x ?xx ?w)
                     (in_y ?yy ?w)

                    (at_lander ?l ?y)
                    (have_soil_analysis ?r ?p) 
                    (visible ?w ?y)
            )
 :effect (and (communicated_soil_data ?p)
  )
)

(:action communicate_image_data
 :parameters (?r - rover ?l - lander ?o - objective ?m - mode ?w - waypoint ?y - waypoint ?x - loc_x ?yy - loc_y)
 :precondition (and (at_x ?x)
				    (at_y ?yy)				
				    (in_x ?x ?w)
				    (in_y ?yy ?w)
                    (at_lander ?l ?y)
                    (have_image ?r ?o ?m)
                    (visible ?w ?y)
            )
 :effect (and 
              (communicated_image_data ?o ?m)
          )
)
 )
