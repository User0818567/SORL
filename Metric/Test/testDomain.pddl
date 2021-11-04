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
		:precondition (and (atMiddleLadder) (keyexist) )
		:effect (and (not (atMiddleLadder) ) (atRightDoor) (increase (quality) 0 ) )
	)
	(:action act2
		:precondition (and (atRightDoor) (keyexist) )
		:effect (and (not (atRightDoor) ) (atMiddleLadder) (increase (quality) 0 ) )
	)
	(:action act3
		:precondition (and (atRightLadder) (keyexist) )
		:effect (and (not (atRightLadder) ) (atMiddleLadder) (increase (quality) 0 ) )
	)
	(:action act4
		:precondition (and (atRightLadder) (keyexist) )
		:effect (and (not (atRightLadder) ) (atLeftLadder) (increase (quality) 0 ) )
	)
	(:action act5
		:precondition (and (atLeftLadder) (keyexist) )
		:effect (and (not (keyexist)) (increase (quality) 100 ) )
	)
	(:action act6
		:precondition (and (atLeftLadder) (keyexist) )
		:effect (and (not (atLeftLadder) ) (atRightLadder) (increase (quality) 0 ) )
	)
	(:action act7
		:precondition (and (atLeftLadder) (not (keyexist)) )
		:effect (and (not (atLeftLadder) ) (atRightLadder) (increase (quality) 0 ) )
	)
	(:action act8
		:precondition (and (atRightLadder) (not (keyexist)) )
		:effect (and (not (atRightLadder) ) (atMiddleLadder) (increase (quality) 0 ) )
	)
	(:action act9
		:precondition (and (atRightLadder) (not (keyexist)) )
		:effect (and (not (atRightLadder) ) (atLeftLadder) (increase (quality) 0 ) )
	)
	(:action act10
		:precondition (and (atMiddleLadder) (not (keyexist)) )
		:effect (and (not (atMiddleLadder) ) (atRightDoor) (increase (quality) 301 ) )
	)
)