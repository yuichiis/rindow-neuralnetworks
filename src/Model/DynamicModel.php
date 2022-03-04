<?php
declare(strict_types=1);
namespace Rindow\NeuralNetworks\Model;

use InvalidArgumentException;
use UnexpectedValueException;
use LogicException;
use ArrayAccess;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\LayerBase;
use Rindow\NeuralNetworks\Gradient\Core\Variable;
use Rindow\NeuralNetworks\Gradient\Core\Undetermined;
use Rindow\NeuralNetworks\Gradient\Core\UndeterminedNDArray;
use Rindow\NeuralNetworks\Gradient\Core\GradientTape;
use Rindow\NeuralNetworks\Gradient\Module;

abstract class DynamicModel extends AbstractModel implements Module
{
    protected $weightVariables = [];
    protected $trainableVariables;
    protected $generation;
    protected $inputsVariables;
    protected $outputsVariables;
    protected $graph = [];
    protected $weights;
    protected $grads;

    /**
    *  @return int
    */
    public function generation() : int
    {
        return $this->generation;
    }
    /**
    *  @return array<Variable>
    */
    public function inputs()
    {
        return $this->inputsVariables;
    }

    /**
    *  @return array<Variable>
    */
    public function outputs()
    {
        return $this->outputsVariables;
    }

    /*
    *  dinamic step interfaces
    */

    /**
    *  @param array<Variable>  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    public function __invoke(...$inputs)
    {
        $outputs = $this->call(...$inputs);
        return $outputs;
    }

    /*
    *  dinamic step interfaces
    */
    /**
    *  @param array<Variable>  $inputs
    *       inputs
    *  @return array<Variable>
    *       outputs
    */
    //protected function call(...$inputs) : array
    //{
    //    throw new LogicException('"call" is not implemented');
    //}

    protected function buildLayers(array $options=null) : void
    {
    //    $model = $this;
    //    $model($x,true)
    }

    protected function getModelGraph()
    {
        if(!isset($this->graph['model'])) {
            $model = $this;
            $func = function($x,$t) use ($model) {
                $training = (GradientTape::$autoBackProp)? true: false;
                return $model($x,$training,$t);
            };
            $options = ['alternateCreator'=>$this];
            //[$weights,$grads] = $this->initWeights();
            //if(count($weights)) {
            //    $options['weights'] = $weights;
            //    $options['grads'] = $grads;
            //}
            $this->graph['model'] = $this->builder->gradient->function($func,$options);
        }
        return $this->graph['model'];
    }

    public function _rawCall($inputs)
    {
        return $this->graph['model']->_rawCall($inputs);
    }

    public function setShapeInspection(bool $enable)
    {
        if($this->shapeInspection==$enable)
            return;
        foreach ($this->submodules() as $module) {
            $module->setShapeInspection($enable);
        }
        $this->shapeInspection = $enable;
    }

    public function submodules() : array
    {
        $modules = [];
        foreach (get_object_vars($this) as $func) {
            if($func instanceof Module) {
                $modules[] = $func;
            }
        }
        return $modules;
    }

    public function variables() : array
    {
        $variables = [];
        foreach ($this->submodules() as $module) {
            $variables = array_merge($variables,$module->variables());
        }
        foreach(get_object_vars($this) as $var) {
            if($var instanceof Variable) {
                $variables[] = $var;
            }
        }

        return $variables;
    }

    public function trainableVariables() : array
    {
        return $this->variables();
    }

    public function reverseSyncWeightVariables() : void
    {
    }

    public function backward(array $dOutputs, ArrayAccess $grads=null, array $oidsToCollect=null) : array
    {
        return $this->graph['model']->backward($dOutputs, $grads, $oidsToCollect);
    }

    protected function trainStep($inputs, $trues)
    {
        $K = $this->backend;
        $nn = $this->builder;
        $g = $nn->gradient();
        $x = $g->Variable($inputs);
        $t = $g->Variable($trues);
        $trues = $this->trueValuesFilter($trues);
        $model = $this->getModelGraph();
        $lossfunc = $this->lossFunction;
        [$loss,$preds] = $nn->with($tape=$g->GradientTape(),
            function() use ($K,$model,$lossfunc,$x,$t,$trues) {
                $predicts = $model($x,$t);
                return [$lossfunc($trues,$predicts),$predicts];
            }
        );
        $lossValue = $K->scalar($loss->value());
        if(is_nan($lossValue)) {
            throw new UnexpectedValueException("loss is unexpected value");
        }
        $params = $this->trainableVariables();
        $gradients = $tape->gradient($loss, $params);
        $this->optimizer->update($params, $gradients);

        if(in_array('accuracy',$this->metrics)) {
            //$preds = $this->forwardLastlayer($preds);
            $accuracy = $this->lossFunction->accuracy($trues,$preds->value());
        } else {
            $accuracy = 0;
        }
        return [$lossValue,$accuracy];
    }

    protected function evaluateStep($inputs,$trues)
    {
        $nn = $this->builder;
        $K = $nn->backend();
        $g = $nn->gradient();
        $x = $g->Variable($inputs);
        $t = $g->Variable($trues);
        $trues = $this->trueValuesFilter($trues);
        $model = $this->getModelGraph();
        $lossfunc = $this->lossFunction;
        $predicts = $model($x,$t);
        $loss = $lossfunc($trues,$predicts);
        $loss = $K->scalar($loss->value());
        $accuracy = $this->lossFunction->accuracy($trues,$predicts->value());
        return [$loss,$accuracy];
    }

    protected function predictStep($inputs,$options)
    {
        $nn = $this->builder;
        $g = $nn->gradient();
        $x = $g->Variable($inputs);
        $t = null;
        $model = $this->getModelGraph();
        $predicts = $model($x,$t);
        return $predicts->value();
    }

    public function saveWeights(&$modelWeights,$portable=null) : void
    {
        $K = $this->backend;
        $modelWeights['weights'] = $modelWeights['weights'] ?? [];
        foreach($this->trainableVariables() as $idx => $weights) {
            $param = $weights->value();
            $param=$K->ndarray($param);
            if($portable)
                $param = $this->converPortableSaveMode($param);
            $modelWeights['weights'][$idx] = serialize($param);
        }
        $optimizer = $this->optimizer();
        $modelWeights['optimizer'] = $modelWeights['optimizer'] ?? [];
        foreach ($optimizer->getWeights() as $idx => $weights) {
            $weights=$K->ndarray($weights);
            $modelWeights['optimizer'][$idx] = serialize($weights);
        }
    }

    public function loadWeights($modelWeights) : void
    {
        $K = $this->backend;
        $nn = $this->builder;
        $g = $nn->gradient();
        $x = new Undetermined();
        $t = new Undetermined();
        $model = $this;
        $lossfunc = $this->lossFunction;
        //[$loss,$preds] = $nn->with($tape=$g->GradientTape(),
        //    function() use ($K,$model,$lossfunc,$x,$t) {
        //        $predicts = $model($x,true,$t);
        //        return [$lossfunc($t,$predicts),$predicts];
        //    }
        //);
        //foreach($this->trainableVariables() as $idx => $weights) {
        //    $param = $weights->value();
        //    $data = unserialize($modelWeights['weights'][$idx]);
        //    $data = $K->array($data);
        //    $K->copy($data,$param);
        //}
        foreach($this->trainableVariables() as $idx => $weights) {
            $data = unserialize($modelWeights['weights'][$idx]);
            $data = $K->array($data);
            $weights->assign($K->copy($data));
        }
        $stack = [$this];
        while($module=array_pop($stack)) {
            $module->reverseSyncWeightVariables();
            foreach($module->submodules() as $m) {
                array_push($stack,$m);
            }
        }
        $optimizer = $this->optimizer();
        $optimizer->build($this->params());
        foreach ($optimizer->getWeights() as $idx => $weights) {
            $data = unserialize($modelWeights['optimizer'][$idx]);
            $data = $K->array($data);
            $K->copy($data,$weights);
        }
    }

    public function summary()
    {
        throw new LogicException('Unsupported function');
    }

    public function __clone()
    {
        throw new LogicException('Dynamic Models cannot be cloned');
    }

    //public function save($filepath,$portable=null) : void
    //{
    //    throw new LogicException('"Unsupported function');
    //}
}
