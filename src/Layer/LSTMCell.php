<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;

class LSTMCell extends AbstractRNNCell
{
    protected int $units;
    //protected $ac;
    protected object $ac_i;
    protected object $ac_f;
    protected object $ac_c;
    protected object $ac_o;

    //protected $inputs;

    /**
     * @param array<int> $input_shape
     */
    public function __construct(
        object $backend,
        int $units,
        ?array $input_shape=null,
        string|object|null $activation=null,
        string|object|null $recurrent_activation=null,
        ?bool $use_bias=null,
        string|callable|null $kernel_initializer=null,
        string|callable|null $recurrent_initializer=null,
        string|callable|null $bias_initializer=null,
        ?bool $unit_forget_bias=null,
    )
    {
        $input_shape = $input_shape ?? null;
        $activation = $activation ?? 'tanh';
        $recurrent_activation = $recurrent_activation ?? 'sigmoid';
        $use_bias = $use_bias ?? true;
        $kernel_initializer = $kernel_initializer ?? 'glorot_uniform';
        $recurrent_initializer = $recurrent_initializer ?? 'orthogonal';
        $bias_initializer = $bias_initializer ?? 'zeros';
        $unit_forget_bias = $unit_forget_bias ?? true;
        //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
        //'activity_regularizer'=null,
        //'kernel_constraint'=null, 'bias_constraint'=null,
        
        parent::__construct($backend);
        $K = $backend;
        $this->units = $units;
        $this->inputShape = $input_shape;
        if($use_bias) {
            $this->useBias = $use_bias;
        }
        $this->setActivation($activation);
        $this->setRecurrentActivation($recurrent_activation);
        $this->setKernelInitializer(
            $kernel_initializer,
            $recurrent_initializer,
            $bias_initializer,
        );
    }

    public function build(mixed $inputShape=null, ?array $sampleWeights=null) : void
    {
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;
        $recurrentInitializer = $this->recurrentInitializer;
        $biasInitializer = $this->biasInitializer;

        $inputShape = $this->normalizeCellInputShape($inputShape);
        //if(count($inputShape)!=1) {
        //    throw new InvalidArgumentException(
        ///        'Unsuppored input shape: ['.implode(',',$inputShape).']');
        //}
        $shape = $inputShape;
        $inputDim = array_pop($shape);
        if($this->kernel===null) {
            if($sampleWeights) {
                $this->kernel = $sampleWeights[0];
                $this->recurrentKernel = $sampleWeights[1];
                $this->bias = $sampleWeights[2];
            } else {
                $this->kernel = $kernelInitializer([
                    $inputDim,$this->units*4],
                    [$inputDim,$this->units]);
                $this->recurrentKernel = $recurrentInitializer(
                    [$this->units,$this->units*4],
                    [$this->units,$this->units]);
                if($this->useBias) {
                    $this->bias = $biasInitializer([$this->units*4]);
                }
            }
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        $this->dRecurrentKernel = $K->zerosLike($this->recurrentKernel);
        if($this->useBias) {
            $this->dBias = $K->zerosLike($this->bias);
        }
        array_push($shape,$this->units);
        $this->outputShape = $shape;
    }

    public function getConfig() : array
    {
        return [
            'units' => $this->units,
            'options' => [
                'input_shape'=>$this->inputShape,
                'use_bias'=>$this->useBias,
                'activation'=>$this->activationName,
                'recurrent_activation'=>$this->recurrentActivationName,
                'kernel_initializer' => $this->kernelInitializerName,
                'recurrent_initializer' => $this->recurrentInitializerName,
                'bias_initializer' => $this->biasInitializerName,
            ]
        ];
    }

    protected function call(
        NDArray $inputs,
        array $states,
        ?bool $training=null,
        ?object $calcState=null,
        ) : array
    {
        $K = $this->backend;
        $prev_h = $states[0];
        $prev_c = $states[1];

        if($this->useBias){
            $outputs = $K->batch_gemm($inputs, $this->kernel,1.0,1.0,$this->bias);
        } else {
            $outputs = $K->gemm($inputs, $this->kernel);
        }
        $outputs = $K->gemm($prev_h, $this->recurrentKernel,1.0,1.0,$outputs);

        $x_i = $K->slice($outputs,
            [0,0],[-1,$this->units]);
        $x_f = $K->slice($outputs,
            [0,$this->units],[-1,$this->units]);
        $x_c = $K->slice($outputs,
            [0,$this->units*2],[-1,$this->units]);
        $x_o = $K->slice($outputs,
            [0,$this->units*3],[-1,$this->units]);

        if($this->activation){
            $calcState->ac_c = new \stdClass();
            $x_c = $this->activation->forward($calcState->ac_c,$x_c,$training);
        }
        if($this->recurrentActivation){
            $calcState->ac_i = new \stdClass();
            $x_i = $this->recurrentActivation->forward($calcState->ac_i,$x_i,$training);
            $calcState->ac_f = new \stdClass();
            $x_f = $this->recurrentActivation->forward($calcState->ac_f,$x_f,$training);
            $calcState->ac_o = new \stdClass();
            $x_o = $this->recurrentActivation->forward($calcState->ac_o,$x_o,$training);
        }
        $next_c = $K->add($K->mul($x_f,$prev_c),$K->mul($x_i,$x_c));
        $ac_next_c = $next_c;
        if($this->activation){
            $calcState->ac = new \stdClass();
            $ac_next_c = $this->activation->forward($calcState->ac,$ac_next_c,$training);
        }
        // next_h = o * ac_next_c
        $next_h = $K->mul($x_o,$ac_next_c);

        $calcState->inputs = $inputs;
        $calcState->prev_h = $prev_h;
        $calcState->prev_c = $prev_c;
        $calcState->x_i = $x_i;
        $calcState->x_f = $x_f;
        $calcState->x_c = $x_c;
        $calcState->x_o = $x_o;
        $calcState->ac_next_c = $ac_next_c;

        return [$next_h,$next_c];
    }

    protected function differentiate(
        array $dStates,
        object $calcState
        ) : array
    {
        $K = $this->backend;
        $dNext_h = $dStates[0];
        $dNext_c = $dStates[1];

        // this merging move to rnnBackward in backend.
        // $dNext_h = $K->add($dOutputs,$dNext_h);

        $dAc_next_c = $K->mul($calcState->x_o,$dNext_h);
        if($this->activation){
            $dAc_next_c = $this->activation->backward($calcState->ac,$dAc_next_c);
        }
        $dNext_c = $K->add($dNext_c, $dAc_next_c);

        $dPrev_c = $K->mul($dNext_c, $calcState->x_f);

        $dx_i = $K->mul($dNext_c,$calcState->x_c);
        $dx_f = $K->mul($dNext_c,$calcState->prev_c);
        $dx_o = $K->mul($dNext_h,$calcState->ac_next_c);
        $dx_c = $K->mul($dNext_c,$calcState->x_i);

        if($this->recurrentActivation){
            $dx_i = $this->recurrentActivation->backward($calcState->ac_i,$dx_i);
            $dx_f = $this->recurrentActivation->backward($calcState->ac_f,$dx_f);
            $dx_o = $this->recurrentActivation->backward($calcState->ac_o,$dx_o);
        }
        if($this->activation){
            $dx_c = $this->activation->backward($calcState->ac_c,$dx_c);
        }

        $dOutputs = $K->stack([$dx_i,$dx_f,$dx_c,$dx_o],axis:1);
        $shape = $dOutputs->shape();
        $batches = array_shift($shape);
        $dOutputs = $dOutputs->reshape([$batches,array_product($shape)]);

        $K->gemm(
            $calcState->prev_h,
            $dOutputs,
            beta:1.0,
            c:$this->dRecurrentKernel,
            transA:true,
        );
        $K->gemm(
            $calcState->inputs,
            $dOutputs,
            beta:1.0,
            c:$this->dKernel,
            transA:true
        );

        if($this->useBias) {
            $K->update_add($this->dBias,$K->sum($dOutputs, axis:0));
        }

        $dInputs = $K->gemm($dOutputs, $this->kernel, transB:true);
        $dPrev_h = $K->gemm($dOutputs, $this->recurrentKernel, transB:true);

        return [$dInputs,[$dPrev_h, $dPrev_c]];
    }
}
