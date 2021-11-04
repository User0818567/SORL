(define (domain Taxi)
(:requirements :typing :fluents)
(:types car region person - object)
(:predicates(at ?p - person ?r - region)
            (in ?p - person ?c - car)
)
(:functions (car-x ?c - car)
            (car-y ?c - car)
            (region-x ?r - region)
            (region-y ?r - region)
			(height ?r - region)
			(width ?r - region)
            (distance)
            )


(:action take-off
 :parameters (?c - car ?r - region ?p - person)
 :precondition (and (in ?p ?c)
                    (>= (car-x ?c) (region-x ?r))
                    (<= (car-x ?c) (+ (region-x ?r) (width ?r)))
                    (>= (car-y ?c) (region-y ?r))
                    (<= (car-y ?c) (+ (region-y ?r) (height ?r)))
                 )
 :effect (and (not (in ?p ?c))
              (at ?p ?r)
              )
 )

(:action take-on
 :parameters (?c - car ?r - region ?p - person)
 :precondition (and (not (in ?p ?c))
                    (at ?p ?r)
 					(>= (car-x ?c) (region-x ?r))
 					(<= (car-x ?c) (+ (region-x ?r) (width ?r)))
 					(>= (car-y ?c) (region-y ?r))
 					(<= (car-y ?c) (+ (region-y ?r) (height ?r)))
                 )
 :effect (and (in ?p ?c)
              (not (at ?p ?r))
              )
 )

(:action glide-right-up
    :parameters (?c - car)
    :effect (and
        (increase (car-x ?c) 2)
        (increase (car-y ?c) 2)
        (increase (distance) 28.2842712)
    )
)

(:action glide-right-down
    :parameters (?c - car)
    :effect (and
        (increase (car-x ?c) 2)
        (decrease (car-y ?c) 2)
        (increase (distance) 28.2842712)
    )
)
(:action glide-left-up
    :parameters (?c - car)
    :effect (and
        (decrease (car-x ?c) 2)
        (increase (car-y ?c) 2)
        (increase (distance) 28.2842712)
    )
)
(:action glide-left-down
    :parameters (?c - car)
    :effect (and
        (decrease (car-x ?c) 2)
        (decrease (car-y ?c) 2)
        (increase (distance) 28.2842712)
    )
)
(:action glide-right
    :parameters (?c - car)
    :effect (and
        (increase (car-x ?c) 2)
        (increase (distance) 20)
    )
)
(:action glide-left
    :parameters (?c - car)
    :effect (and
        (decrease (car-x ?c) 2)
        (increase (distance) 20)
    )
)
(:action glide-up
    :parameters (?c - car)
    :effect (and
        (increase (car-y ?c) 2)
        (increase (distance) 20)
    )
)
(:action glide-down
    :parameters (?c - car)
    :effect (and
        (decrease (car-y ?c) 2)
        (increase (distance) 20)
    )
)





)

