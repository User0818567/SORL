(define (domain Rover)
(:requirements :typing :fluents)
(:types rover waypoint store camera mode lander objective)

(:predicates 
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

(:functions (energy ?r - rover) 
            (recharges)
            (waypoint_x ?w - waypoint)
            (waypoint_y ?w - waypoint)
            (waypoint_width ?w - waypoint)
            (waypoint_height ?w - waypoint)
            (rover_x ?r - rover)
            (rover_y ?r - rover)
            (total_time)
            (distance)
             )
	
(:action navigate-right
:parameters (?x - rover) 
:precondition (and (>= (energy ?x) 0)
	    )
:effect (and (decrease (energy ?x) 0.05) 
             (increase (rover_x ?x) 0.05)
             (increase (total_time) 0.05)
             (increase (distance) 0.05)
		)
)

(:action navigate-left
:parameters (?x - rover) 
:precondition (and (>= (energy ?x) 0)
      )
:effect (and (decrease (energy ?x) 0.05) 
             (decrease (rover_x ?x) 0.05)
             (increase (total_time) 0.05)
             (increase (distance) 0.05)
    )
)

(:action navigate-up
:parameters (?x - rover) 
:precondition (and (>= (energy ?x) 0)
      )
:effect (and (decrease (energy ?x) 0.05)
             (increase (rover_y ?x) 0.05)
             (increase (total_time) 0.05)
             (increase (distance) 0.05)
    )
)

(:action navigate-down
:parameters (?x - rover) 
:precondition (and (>= (energy ?x) 0)
      )
:effect (and (decrease (energy ?x) 0.05)
             (increase (rover_y ?x) 0.05)
             (increase (total_time) 0.05)
             (increase (distance) 0.05)
    )
)


(:action navigate-left-down
:parameters (?x - rover) 
:precondition (and (>= (energy ?x) 0)
      )
:effect (and (decrease (energy ?x) 0.05) 
             (decrease (rover_x ?x) 0.05)
             (decrease (rover_y ?x) 0.05)
             (increase (total_time) 0.05)
             (increase (distance) 0.07071)
    )
)


(:action navigate-left-up
:parameters (?x - rover) 
:precondition (and (>= (energy ?x) 0)
      )
:effect (and (decrease (energy ?x) 0.05) 
             (decrease (rover_x ?x) 0.05)
             (increase (rover_y ?x) 0.05)
             (increase (total_time) 0.05)
             (increase (distance) 0.07071)
    )
)

(:action navigate-right-down
:parameters (?x - rover) 
:precondition (and (>= (energy ?x) 0)
      )
:effect (and (decrease (energy ?x) 0.05) 
             (increase (rover_x ?x) 0.05)
             (decrease (rover_y ?x) 0.05)
             (increase (total_time) 0.05)
             (increase (distance) 0.07071)
    )
)

(:action navigate-right-up
:parameters (?x - rover) 
:precondition (and (>= (energy ?x) 0)
      )
:effect (and (decrease (energy ?x) 0.05) 
             (increase (rover_x ?x) 0.05)
             (increase (rover_y ?x) 0.05)
             (increase (total_time) 0.05)
             (increase (distance) 0.07071)
    )
)


(:action recharge
:parameters (?x - rover ?w - waypoint)
:precondition (and 
                  (<= (energy ?x) 8)
                  (<= (rover_x ?x) (+ (waypoint_x ?w)(waypoint_width ?w)))
                  (>= (rover_x ?x) (waypoint_x ?w))
                  (<= (rover_y ?x) (+ (waypoint_y ?w)(waypoint_height ?w)))
                  (>= (rover_y ?x) (waypoint_y ?w))
                  (in_sun ?w)
                  )
:effect (and 
            (increase (energy ?x) 0.05)
            (increase (recharges) 1)) 
)

(:action sample_soil
:parameters (?x - rover ?s - store ?w - waypoint)
:precondition (and (<= (rover_x ?x) (+ (waypoint_x ?w)(waypoint_width ?w)))
                   (>= (rover_x ?x) (waypoint_x ?w))
                   (<= (rover_y ?x) (+ (waypoint_y ?w)(waypoint_height ?w)))
                   (>= (rover_y ?x) (waypoint_y ?w))

                   (>= (energy ?x) 0.3)

                   (at_soil_sample ?w) 
                   (equipped_for_soil_analysis ?x) 
                   (store_of ?s ?x) 
                   (empty ?s)
		)
:effect (and (not (empty ?s)) 
                  (full ?s) 
                  (decrease (energy ?x) 0.3) 
                  (have_soil_analysis ?x ?w) 
                  (not (at_soil_sample ?w))
		)
)

(:action sample_rock
:parameters (?x - rover ?s - store ?w - waypoint)
:precondition (and  (<= (rover_x ?x) (+ (waypoint_x ?w)(waypoint_width ?w)))
                    (>= (rover_x ?x) (waypoint_x ?w))
                    (<= (rover_y ?x) (+ (waypoint_y ?w)(waypoint_height ?w)))
                    (>= (rover_y ?x) (waypoint_y ?w))

                    (>= (energy ?x) 0.5)
                    (at_rock_sample ?w)
                    (equipped_for_rock_analysis ?x)
                    (store_of ?s ?x)
                    (empty ?s)
		)
:effect (and (not (empty ?s))
             (full ?s)
             (decrease (energy ?x) 0.5)
             (have_rock_analysis ?x ?w)
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
 :parameters (?r - rover ?i - camera ?t - objective ?w - waypoint)
 :precondition (and (equipped_for_imaging ?r)
                    (>= (energy ?r) 0.2)
                    (calibration_target ?i ?t)
                    (<= (rover_x ?r) (+ (waypoint_x ?w)(waypoint_width ?w)))
                    (>= (rover_x ?r) (waypoint_x ?w))
                    (<= (rover_y ?r) (+ (waypoint_y ?w)(waypoint_height ?w)))
                    (>= (rover_y ?r) (waypoint_y ?w))
                    (visible_from ?t ?w)
                    (on_board ?i ?r)
		)
 :effect (and (decrease (energy ?r) 0.2)
              (calibrated ?i ?r) )
)

(:action take_image
 :parameters (?r - rover ?w - waypoint ?o - objective ?i - camera ?m - mode)
 :precondition (and (calibrated ?i ?r)
             			  (on_board ?i ?r)
                    (equipped_for_imaging ?r)
                    (supports ?i ?m)
			              (visible_from ?o ?w)
                    (<= (rover_x ?r) (+ (waypoint_x ?w)(waypoint_width ?w)))
                    (>= (rover_x ?r) (waypoint_x ?w))
                    (<= (rover_y ?r) (+ (waypoint_y ?w)(waypoint_height ?w)))
                    (>= (rover_y ?r) (waypoint_y ?w))
			              (>= (energy ?r) 0.1)
               )
 :effect (and (have_image ?r ?o ?m)
              (not (calibrated ?i ?r))
              (decrease (energy ?r) 0.1)
		)
)

(:action communicate_soil_data
 :parameters (?r - rover ?l - lander ?p - waypoint ?w - waypoint ?y - waypoint)
 :precondition (and 
                    (<= (rover_x ?r) (+ (waypoint_x ?w)(waypoint_width ?w)))
                    (>= (rover_x ?r) (waypoint_x ?w))
                    (<= (rover_y ?r) (+ (waypoint_y ?w)(waypoint_height ?w)))
                    (>= (rover_y ?r) (waypoint_y ?w))

                    (at_lander ?l ?y)
                    (have_soil_analysis ?r ?p) 
                    (visible ?w ?y)
                    (>= (energy ?r) 0.4)
            )
 :effect (and (communicated_soil_data ?p)
              (decrease (energy ?r) 0.4)
	)
)

(:action communicate_rock_data
 :parameters (?r - rover ?l - lander ?p - waypoint ?w - waypoint ?y - waypoint)
 :precondition (and (<= (rover_x ?r) (+ (waypoint_x ?w)(waypoint_width ?w)))
                    (>= (rover_x ?r) (waypoint_x ?w))
                    (<= (rover_y ?r) (+ (waypoint_y ?w)(waypoint_height ?w)))
                    (>= (rover_y ?r) (waypoint_y ?w))

                    (at_lander ?l ?y)
                    (have_rock_analysis ?r ?p)
                    (>= (energy ?r) 0.4)
                    (visible ?w ?y)
            )
 :effect (and   
                (communicated_rock_data ?p)
                (decrease (energy ?r) 0.4)
          )
)


(:action communicate_image_data
 :parameters (?r - rover ?l - lander ?o - objective ?m - mode ?w - waypoint ?y - waypoint)
 :precondition (and (<= (rover_x ?r) (+ (waypoint_x ?w)(waypoint_width ?w)))
                    (>= (rover_x ?r) (waypoint_x ?w))
                    (<= (rover_y ?r) (+ (waypoint_y ?w)(waypoint_height ?w)))
                    (>= (rover_y ?r) (waypoint_y ?w))
                    (at_lander ?l ?y)
                    (have_image ?r ?o ?m)
                    (visible ?w ?y)
                    (>= (energy ?r) 0.6)
            )
 :effect (and 
              (communicated_image_data ?o ?m)
              (decrease (energy ?r) 0.6)
          )
)

)
