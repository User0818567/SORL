(define
    (domain Montezumatest)
    (:requirements :typing :fluents)
    (:predicates
        (keyexist)
        (atRightDoor)
        (atKey)
        (atMiddleLadder)
        (atLeftLadder)
        (atRightLadder)
        (atDown1)
        (atDown2)
    )
    (:functions (quality))
    	(:action act0
		:precondition (and (atMiddleLadder) (keyexist) )
		:effect (and (not (atMiddleLadder) ) (atRightLadder) (increase (quality) 0 ) )
	)
	(:action act1
		:precondition (and (atRightLadder) (keyexist) )
		:effect (and (not (atRightLadder) ) (atDown2) (increase (quality) 0 ) )
	)
	(:action act2
		:precondition (and (atDown2) (keyexist) )
		:effect (and (not (atDown2) ) (atRightLadder) (increase (quality) 0 ) )
	)
	(:action act3
		:precondition (and (atDown2) (keyexist) )
		:effect (and (not (atDown2) ) (atDown1) (increase (quality) 1 ) )
	)
	(:action act4
		:precondition (and (atRightLadder) (keyexist) )
		:effect (and (not (atRightLadder) ) (atMiddleLadder) (increase (quality) 0 ) )
	)
	(:action act5
		:precondition (and (atDown1) (keyexist) )
		:effect (and (not (atDown1) ) (atLeftLadder) (increase (quality) 0 ) )
	)
	(:action act6
		:precondition (and (atLeftLadder) (keyexist) )
		:effect (and (not (atLeftLadder) ) (atDown1) (increase (quality) 5 ) )
	)
	(:action act7
		:precondition (and (atMiddleLadder) (keyexist) )
		:effect (and (not (atMiddleLadder) ) (atDown1) (increase (quality) 0 ) )
	)
	(:action act8
		:precondition (and (atDown1) (keyexist) )
		:effect (and (not (atDown1) ) (atDown2) (increase (quality) 0 ) )
	)
)