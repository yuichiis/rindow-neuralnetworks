<?php
declare(strict_types=1);

try {
    var_dump(0);
} catch(Ex1 | Ex2 $e) {
    echo $e->getMessage();
    throw new Exception("Error Processing Request", 1);
} finally {
    var_dump(1);
}
echo "split in foreach\n";
$ziped = [[1,2],[3,4]];
foreach($ziped as [$a,$b]) {
    echo "$a,$b\n";
}
echo "Anonymous class\n";
$a = new class {
    public function hello($var) : string
    {
        return "hello $var";
    }
};
echo $a->hello('foo')."\n";
echo "split map\n";
$a = ['a'=>'A','b'=>'B'];
$b = new stdClass();
['a' => $b->a, 'b' => $b->b] = $a;
var_dump($b);
// php7.2
echo "Object typehint\n";
$a = function(object $a) { return $a; };
var_dump($a(new stdClass()));
echo "Abstract method overriding\n";
abstract class AbsLatcBase { abstract public function f(string $a);}
abstract class AbsLatcSub extends AbsLatcBase { abstract public function f($a) : int;}
// ***** PHP Fatal error:  Declaration of AbsLatcSubX::f(int $a) must be compatible with AbsLatcBaseX::f($a): string 
//abstract class AbsLatcBaseX { abstract public function f($a) : string;}
//abstract class AbsLatcSubX extends AbsLatcBaseX { abstract public function f(int $a);}
echo "spl_object_id()\n";
var_dump(spl_object_id(new stdClass()));
// php7.3
echo "Array Destructuring supports Reference Assignments\n";
$a = ['A','B'];
[ &$aa, $bb] = $a;
$aa = 'UPDATE';
var_dump($a);
echo "array_key_first(),array_key_last()\n";
$a = ['a'=>'A','b'=>'B'];
var_dump(array_key_first($a));
var_dump(array_key_last($a));
echo "hrtime()\n";
$a = hrtime(true);
var_dump(hrtime(true)-$a);
// php7.4
echo "Typed properties\n";
class TypedProp {
    public int $id;
    public string $name;
}
echo "Arrow functions\n";
$factor = 10;
$nums = array_map(fn($n) => $n * $factor, [1, 2, 3, 4]);
var_dump($nums);
echo "Limited return type covariance\n";
// *** But Arg types are NOT applicable ***
class LatcA {};
class LatcB extends LatcA{};
class LatcBase { public function f($a) : LatcA {}}
class LatcSub extends LatcBase { public function f($a) : LatcB {}}
echo "Unpacking Inside Arrays\n";
$a = ['a','b'];
$b = ['c', ...$a];
var_dump($b);
echo "WeakReference\n";
$a = new stdClass();
$b = WeakReference::create($a);
var_dump($b->get());
var_dump(spl_object_id($a));
unset($a);
$a = new stdClass();
var_dump(spl_object_id($a));
var_dump($b->get());

// php8.0
echo "WeakMap\n";
$a = new stdClass();
$b = new WeakMap();
$b[$a] = 'A';
var_dump(count($b));
unset($a);
var_dump(count($b));
echo "constructor property promotion\n";
class ConProp {
    public function __construct(protected int $x, protected int $y = 0) {
    }
}
var_dump(new ConProp(1));
echo "Named Arguments\n";
function fmain(...$args) {
    echo "==args==\n";
    var_dump($args);
    echo "=======\n";
    func(...$args);
}

function func(array $inputs, ?int $a=null, $b=null, bool $training=null)
{
    var_dump($training);
}

func([]);
func([],training:true);
fmain([],training:true);
fmain([],1,training:true);

$a = [];
$b = array_shift($a);
var_dump($a);
var_dump($b);
$reffunc = new ReflectionFunction('func');
$parameters = $reffunc->getParameters();
echo "====function args====\n";
foreach($parameters as $parameter) {
    echo $parameter->getName()." #".$parameter->getPosition().": ";
    if($parameter->allowsNull()) {
        //echo "?";
    }
    if($parameter->hasType()) {
        echo $parameter->getType();
    }
    if($parameter->isOptional()) {
        echo "(optional)";
    }
    echo "\n";
}

echo "==== object in_array====\n";
$a = new stdClass();
$b = new stdClass();
$c = (object)['C'];
$d = new stdClass();
$array = [$a,$b,$c];
var_dump(in_array($a,$array));
var_dump(in_array($c,$array));
var_dump(in_array($d,$array));
var_dump(in_array($a,$array,true));
var_dump(in_array($d,$array,true));

//[$a, ...$b] = [1,2,3]; // Error
//var_dump($a);
//var_dump($b);
$a = 1;
$b = [2,3];
$c = [$a, ...$b];
var_dump($c);
exit();
