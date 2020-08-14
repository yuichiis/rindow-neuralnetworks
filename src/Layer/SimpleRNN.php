<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class SimpleRNN extends AbstractRNNLayer
{
    use GenericUtils;
    protected $backend;
    protected $units;
    protected $activation;
    protected $useBias;
    protected $kernelInitializerName;
    protected $recurrentInitializerName;
    protected $biasInitializerName;
    protected $returnSequence;
    protected $returnState;
    protected $goBackward;
    protected $statefull;
    protected $cell;
    protected $timesteps;
    protected $feature;

    protected $calcStates;
    protected $initialStates;
    protected $origInputsShape;
    
    public function __construct($backend,int $units, array $options=null)
    {
        extract($this->extractArgs([
            'input_shape'=>null,
            'activation'=>'tanh',
            'use_bias'=>true,
            'kernel_initializer'=>'sigmoid_normal',
            'recurrent_initializer'=>'sigmoid_normal',
            'bias_initializer'=>'zeros',
            'return_sequence'=>false,
            'return_state'=>false,
            'go_backward'=>false,
            'stateful'=>false,
            //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
            //'activity_regularizer'=null,
            //'kernel_constraint'=null, 'bias_constraint'=null,
        ],$options));
        $this->backend = $K = $backend;
        $this->units = $units;
        $this->inputShape = $input_shape;
        if($use_bias) {
            $this->useBias = $use_bias;
        }
        $this->setActivation($activation);
        $activation = $this->activation;
        $this->activation = null;
        $this->kernelInitializerName = $kernel_initializer;
        $this->recurrentInitializerName = $recurrent_initializer;
        $this->biasInitializerName = $bias_initializer;
        $this->returnSequence=$return_sequence;
        $this->returnState = $return_state;
        $this->goBackward = $go_backward;
        $this->stateful = $stateful;
        $this->cell = new SimpleRNNqCell(
            $this->backend,
            $this->units,
            [
            'activation'=>$activation,
            'use_bias'=>$this->useBias,
            'kernel_initializer'=>$this->kernelInitializerName,
            'recurrent_initializer'=>$this->recurrentInitializerName,
            'bias_initializer'=>$this->biasInitializerName,
            ]);
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
        $K = $this->backend;

        $inputShape = $this->normalizeInputShape($inputShape);
        //if(count($inputShape)!=1) {
        //    throw new InvalidArgumentException(
        ///        'Unsuppored input shape: ['.implode(',',$inputShape).']');
        //}
        if(count($inputShape)!=2){
            throw new InvalidArgumentException('Unsuppored input shape.:['.implode(',',$inputShape).']');
        }
        $this->timesteps = $inputShape[0];
        $this->feature = $inputShape[1];
        $this->cell->build([$this->feature],$options);
        if($this->returnSequence){
            $this->outputShape = [$this->timesteps,$this->units];
        }else{
            $this->outputShape = [$this->units];
        }
        return $this->outputShape;
    }

    public function getParams() : array
    {
        return $this->cell->getParams();
    }

    public function getGrads() : array
    {
        return $this->cell->getGrads();
    }

    public function getConfig() : array
    {
        return [
            'units' => $this->units,
            'options' => [
                'input_shape'=>$this->inputShape,
                'use_bias'=>$this->useBias,
                'activation'=>$this->activationName,
                'kernel_initializer' => $this->kernelInitializerName,
                'recurrent_initializer' => $this->recurrentInitializerName,
                'bias_initializer' => $this->biasInitializerName,
                'return_sequence'=>$this->returnSequence,
                'return_state'=>$this->returnState,
                'go_backward'=>$this->goBackward,
                'stateful'=>$this->stateful,
            ]
        ];
    }

    protected function call(NDArray $inputs,bool $training, array $initalStates=null, array $options=null)
    {
        $K = $this->backend;
        [$batches,$timesteps,$feature]=$inputs->shape();
        if($initialStates===null&&
            $this->stateful) {
            $initialStates = $this->initialStates;
        }
        if($initialStates===null){
            $initialStates = [$K->zeros([$batches,$this->units])];
        }
        $outputs = null;
        if($this->returnSequence){
            $outputs=$K->zeros([$batches,$timesteps,$this->units]);
        }
        [$outputs,$states,$calcStates] = $K->rnn(
            [$this->cell,'forward'],
            $inputs,
            $initialStates,
            $training,
            $outputs,
            $this->goBackward
        );
        $this->calcStates = $calcStates;
        $this->origInputsShape = $inputs->shape();
        if($this->stateful) {
            $this->initialStates = $states;
        }
        if($this->returnState){
            return [$outputs,$states];
        } else {
            return $outputs;
        }
    }

    protected function differentiate(NDArray $dOutputs, array $dNextStates=null)
    {
        $K = $this->backend;
        $dInputs=$K->zeros($this->origInputsShape);
        if($dNextStates===null){
            $dNextStates = [$K->zeros([$this->origInputsShape[0],$this->units])];
        }

        $grads = $this->cell->getGrads();
        foreach($grads as $grad){
            $K->clear($grad);
        }
        [$dInputs,$dPrevStates] = $K->rnnBackward(
            [$this->cell,'backward'],
            $dOutputs,
            $dNextStates,
            $this->calcStates,
            $dInputs,
            $this->goBackward
        );
        $this->calcStates = null;
        if($this->returnState) {
            return [$dInputs, $dPrevStates];
        } else {
            return $dInputs;
        }
    }
}
